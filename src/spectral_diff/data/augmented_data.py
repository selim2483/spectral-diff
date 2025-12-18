import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from lightning.pytorch import LightningDataModule

from utils.image import center_image

DEFAULT_SCALE = (.25, .75)
DEFAULT_ANGLE = (0., 2 * math.pi)
DEFAULT_CROP_SIZE = 256

def augment(real,scale,theta):
    if real.ndim == 3:
        real = real.unsqueeze(0)
    aff = torch.stack(
        (torch.stack((torch.cos(theta) * scale, -torch.sin(theta) * scale, 0 * scale), dim=1),
        torch.stack((torch.sin(theta) * scale, torch.cos(theta) * scale, 0 * scale), dim=1)),
        dim=1)
    grid = F.affine_grid(aff, real.size())
    t_real = F.grid_sample(real, grid)
    t_real = TF.center_crop(t_real, 256)
    return t_real.squeeze()

class Augment(torch.nn.Module):
    def __init__(self, scale, angle):
        super().__init__()
        self.min_scale, self.max_scale = scale
        self.min_angle, self.max_angle = angle
    
    def forward(self, x: torch.Tensor):
        scale = torch.rand(1) * (self.max_scale - self.min_scale) + self.min_scale
        angle = torch.rand(1) * (self.max_angle - self.min_angle) + self.min_angle
        return augment(x, scale, angle)
    
class AugmentedSet(Dataset):
    def __init__(
            self, root: Path, 
            scale: Sequence[float] = DEFAULT_SCALE, 
            angle: Sequence[float] = DEFAULT_ANGLE, 
            resolution: int = DEFAULT_CROP_SIZE, 
            mode: str = 'train'
        ):
        self.root = root
        self.scale = scale
        self.angle = angle
        self.resolution = resolution
        self.mode = mode

        self.image_paths = list(self.root.iterdir())

        if self.mode == 'train':
            crop_transform = transforms.RandomCrop
        elif self.mode == 'valid':
            crop_transform = transforms.CenterCrop
        
        self.transform = transforms.Compose([
            crop_transform(
                int(math.sqrt(2) * max(self.scale) * self.resolution)),
            transforms.ToTensor(),  
            transforms.Normalize((.5), (.5))
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        img = self.transform(img)
        img, mean = center_image(img)
        return {'images': img, 'mean': mean}
    
class AugmentedDataModule(LightningDataModule):
    def __init__(
            self, 
            root: Path | str,
            scale: Sequence[float] = DEFAULT_SCALE, 
            angle: Sequence[float] = DEFAULT_ANGLE, 
            crop_size: int = DEFAULT_CROP_SIZE,  
            batch_size: int | None = 1, 
            num_workers: int = 1,
        ):
        super().__init__()
        self.save_hyperparameters()
    
    def train_dataloader(self):
        dataset = AugmentedSet(
            self.hparams.root / 'train', self.hparams.scale, 
            self.hparams.angle, self.hparams.crop_size, mode='train')
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, shuffle=True)
    
    def val_dataloader(self):
        dataset = AugmentedSet(
            self.hparams.root / 'test', self.hparams.scale, 
            self.hparams.angle, self.hparams.crop_size, mode='valid')
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, shuffle=False)