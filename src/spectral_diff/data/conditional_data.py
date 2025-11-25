from pathlib import Path
import random
from typing import Sequence
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision.transforms import transforms
from lightning.pytorch import LightningDataModule

from utils.image import imread


def load_cloudy_image(path: Path):
    image = torch.load(path).squeeze()
    image = torch.flip(image, dims=(1,))
    image = image.transpose(0,1)
    return image

class CloudyTrainSet(Dataset):
    def __init__(self, root: Path, nframes: int = None):
        self.root = root
        self.nframes = nframes

        self.image_paths = list(self.root.iterdir())

        self.transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize((.5), (.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = load_cloudy_image(self.image_paths[index])
        nz = image.shape[0]
        idx = random.randint(1, image.shape[0] - 1)

        return {
            'image': self.transform(image[idx]), 
            'condition': self.transform(image[idx - 1])
        }

class CloudyTestSet(Dataset):
    def __init__(self, root: Path, nframes: int = None):
        self.root = root
        self.nframes = nframes

        self.image_paths = list(self.root.iterdir())

        self.transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize((.5), (.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = load_cloudy_image(self.image_paths[index])
        nz = image.shape[0]
        nframes = self.nframes or random.randint(1,nz)
        start_idx = random.randint(0, nz - nframes)

        return {'images': image[start_idx:start_idx+nframes]}
    
class CloudyDataModule(LightningDataModule):
    def __init__(
            self, 
            root: Path | str, 
            nframes: int = None,
            batch_size: int | None = 1, 
            num_workers: int = 1
        ):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self):
        dataset = CloudyTrainSet(Path(self.hparams.root), self.hparams.nframes)
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, shuffle=True
        )
    
    def val_dataloader(self):
        dataset = CloudyTestSet(Path(self.hparams.root), self.hparams.nframes)
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, shuffle=False
        )