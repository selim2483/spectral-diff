from pathlib import Path
import random
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from lightning.pytorch import LightningDataModule

from utils.image import center_image


def load_cloudy_image(path: Path):
    image = torch.load(path).squeeze()
    image = torch.flip(image, dims=(1,))
    image = image.transpose(0,1)
    return image

class CloudyImageSet(Dataset):
    def __init__(
            self, root: Path, crop_size: int, 
            resize_strategy: str = 'random_crop', random: bool = True,
            center_data: bool = True):
        self.root = root
        self.crop_size = crop_size
        self.resize_strategy = resize_strategy
        self.random = random
        self.center_data = center_data

        self.image_paths = list(self.root.iterdir())

        if self.resize_strategy.lower() == 'crop':
            if random:
                resize_transform = transforms.RandomCrop(self.crop_size)
            else:
                resize_transform = transforms.CenterCrop(self.crop_size)
        elif self.resize_strategy.lower() == 'resize':
            resize_transform = transforms.Resize(
                (self.crop_size, self.crop_size))

        self.transform = transforms.Compose([
            resize_transform,  
            transforms.Normalize((.5), (.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = load_cloudy_image(self.image_paths[index])

        if self.random:
            frame_idx = random.randint(0, image.shape[0] - 1)
        else:
            frame_idx = image.shape[0] // 2 

        img = image[frame_idx].unsqueeze(0)
        img = self.transform(img)
        img, mean = center_image(img, center=self.center_data)

        return {'images': img, 'mean': mean}
    
class CloudyHDF5ImageSet(Dataset):
    def __init__(
            self, root: Path, crop_size: int, 
            resize_strategy: str = 'random_crop', random: bool = True,
            center_data: bool = True):
        self.root = root
        self.crop_size = crop_size
        self.resize_strategy = resize_strategy
        self.random = random
        self.center_data = center_data

        self.hf = h5py.File(self.root, 'r')
        # self._cube_sizes = {grp: len(self.hf[grp]) for grp in (self.hf.keys())}

        if self.resize_strategy.lower() == 'crop':
            if random:
                resize_transform = transforms.RandomCrop(self.crop_size)
            else:
                resize_transform = transforms.CenterCrop(self.crop_size)
        elif self.resize_strategy.lower() == 'resize':
            resize_transform = transforms.Resize(
                (self.crop_size, self.crop_size))

        self.transform = transforms.Compose([
            resize_transform,  
            transforms.Normalize((.5), (.5))
        ])

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, index):
        grp = self.hf[index]
        nz = len(grp)
        
        if self.random:
            frame_idx = random.randint(0, nz - 1)
        else:
            frame_idx = nz // 2 

        img = grp[f'frame_{frame_idx:03}'].unsqueeze(0)
        img = self.transform(img)
        img, mean = center_image(img, center=self.center_data)

        return {'images': img, 'mean': mean}
    
class CloudyImageDataModule(LightningDataModule):
    def __init__(
            self, root: Path | str, crop_size: int, 
            resize_strategy: str = 'crop',
            batch_size: int | None = 1, num_workers: int = 1,
            center_data: bool = True
        ):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self):
        dataset = CloudyImageSet(
            root=Path(self.hparams.root), crop_size=self.hparams.crop_size,
            resize_strategy=self.hparams.resize_strategy, random=True, 
            center_data=self.hparams.center_data)
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, shuffle=True
        )
    
    def val_dataloader(self):
        dataset = CloudyImageSet(
            root=Path(self.hparams.root), crop_size=self.hparams.crop_size,
            resize_strategy=self.hparams.resize_strategy, random=False, 
            center_data=self.hparams.center_data)
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, shuffle=False
        )