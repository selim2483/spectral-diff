from pathlib import Path
import random
from typing import Sequence
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision.transforms import transforms
from lightning.pytorch import LightningDataModule

from utils.image import imread

CROP_RATIO = .25
USE_FLIP = True
CENTER_DATA = True
SAMPLE_EVERY_N_STEP = 1000
VALIDATION_BATCH_SIZE = 4
BANDS = list(range(3))

class CropSet(IterableDataset):
    """
    A dataset comprised of crops of a single image or several images.
    """
    def __init__(
            self, image: torch.Tensor, crop_ratio: float = CROP_RATIO, 
            use_flip: bool = USE_FLIP, dataset_size: int = SAMPLE_EVERY_N_STEP, 
            center_data: bool = CENTER_DATA):
        """
        Args:
            image (torch.tensor): The image to generate crops from.
                Can be of shape (C,H,W) or (B,C,H,W) in case of
                several images.
            crop_ratio (tuple(int, int)): The spatial dimensions of
                the crops to be taken.
            use_flip (bool): Wheather to use horizontal flips of the
                image.
            dataset_size (int): The amount of images in a single
                epoch of training. 
                For training datasets, this should be a high number
                to avoid overhead from pytorch_lightning.
        """
        self.crop_size = int(min(image[0].shape[-2:]) * crop_ratio)
        self.crop_size = int(self.crop_size / 2**3) * 2**3
        print('crop_size', self.crop_size)
        self.dataset_size = dataset_size

        transform_list = [transforms.RandomHorizontalFlip()] if use_flip else []
        transform_list += [
            transforms.RandomCrop(
                self.crop_size, pad_if_needed=False, padding_mode='constant')
        ]

        self.transform = transforms.Compose(transform_list)

        if center_data:
            self.mean = image.squeeze(0).mean(dim=(-1,-2), keepdim=True)
        else:
            self.mean = torch.zeros_like(image.squeeze(0))
            
        self.img = image - self.mean
        
    def __iter__(self):
        # If the training is multi-image, choose one of them to get
        # the crop from
        def next_crop():
            while True:
                img = self.img if len(self.img.shape) == 3 else random.choice(self.img)
                img_crop = self.transform(img)
                yield {'images': img_crop, 'mean': self.mean}

        return iter(next_crop())
    
class SingleSet(Dataset):
    """
    A dataset comprised of crops of a single image or several images.
    """
    def __init__(
            self, 
            image: torch.Tensor, 
            dataset_size: int = VALIDATION_BATCH_SIZE, 
            center_data: bool = CENTER_DATA
        ):
        """
        Args:
            image (torch.tensor): The image to generate crops from.
                Can be of shape (C,H,W) or (B,C,H,W) in case of
                several images.
            use_flip (bool): Wheather to use horizontal flips of the
                image.
            dataset_size (int): The amount of images in a single
                epoch of training. 
                For training datasets, this should be a high number
                to avoid overhead from pytorch_lightning.
        """
        self.dataset_size = dataset_size

        if center_data:
            self.mean = image.squeeze(0).mean(dim=(-1,-2), keepdim=True)
        else: 
            self.mean = torch.zeros_like(image.squeeze(0))

        self.img = image - self.mean

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        # If the training is multi-image, choose one of them to get
        # the crop from
        img = self.img if len(self.img.shape) == 3 else random.choice(self.img)
        return {'images': img, 'mean': self.mean}
    
class SingleImageDataModule(LightningDataModule):
    def __init__(
            self, image_path: str | Path, batch_size: int | None = 1, 
            num_workers: int = 1, crop_ratio: float = CROP_RATIO, 
            sample_every_n_steps: int = SAMPLE_EVERY_N_STEP, 
            val_batch_size: int = VALIDATION_BATCH_SIZE, 
            bands: Sequence[int] = BANDS, center_data: bool = CENTER_DATA):
        super().__init__()
        self.save_hyperparameters()
        self.image = imread(self.hparams.image_path)
        self.image = self.image[...,bands,:,:] * 2 - 1
        self.nbands = len(bands)
        self.image_shape = self.image.shape[1:]

    def train_dataloader(self):
        train_size = self.hparams.sample_every_n_steps * self.hparams.batch_size
        train_dataset = CropSet(
            self.image, self.hparams.crop_ratio, use_flip=False, 
            dataset_size=train_size, center_data=self.hparams.center_data)
        return DataLoader(
            train_dataset, batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        val_dataset = SingleSet(
            self.image, dataset_size=self.hparams.val_batch_size, 
            center_data=self.hparams.center_data)
        return DataLoader(
            val_dataset, batch_size=self.hparams.val_batch_size, 
            num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True)