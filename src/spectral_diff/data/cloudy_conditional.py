from pathlib import Path
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from lightning.pytorch import LightningDataModule


def load_cloudy_image(path: Path):
    image = torch.load(path).squeeze()
    image = torch.flip(image, dims=(1,))
    image = image.transpose(0,1)
    return image

class CloudyConditionalTrainSet(Dataset):
    def __init__(self, root: Path, crop_size: int, nframes: int = None):
        self.root = root
        self.nframes = nframes
        self.crop_size = crop_size

        self.image_paths = list(self.root.iterdir())

        self.transform = transforms.Compose([
            transforms.RandomCrop(self.crop_size),
            transforms.Normalize((.5), (.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = load_cloudy_image(self.image_paths[index])
        nz = image.shape[0]
        idx = random.randint(1, image.shape[0] - 1)

        return {
            'image': self.transform(image[idx].unsqueeze(0)), 
            'condition': self.transform(image[idx - 1].unsqueeze(0))
        }

class CloudyConditionalTestSet(Dataset):
    def __init__(self, root: Path, crop_size: int, nframes: int = None):
        self.root = root
        self.nframes = nframes
        self.crop_size = crop_size

        self.image_paths = list(self.root.iterdir())

        self.transform = transforms.Compose([
            transforms.CenterCrop(self.crop_size),  
            transforms.Normalize((.5), (.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = load_cloudy_image(self.image_paths[index])
        nz = image.shape[0]
        nframes = self.nframes or random.randint(1,nz)
        start_idx = random.randint(0, nz - nframes)
        img = self.transform(image[start_idx:start_idx+nframes])
        return {'images': img}
    
class CloudyConditionalDataModule(LightningDataModule):
    def __init__(
            self, 
            root: Path | str, 
            crop_size: int,
            nframes: int = None,
            batch_size: int | None = 1, 
            num_workers: int = 1
        ):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self):
        dataset = CloudyConditionalTrainSet(
            root=Path(self.hparams.root), crop_size=self.hparams.crop_size, 
            nframes=self.hparams.nframes)
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, shuffle=True
        )
    
    def val_dataloader(self):
        dataset = CloudyConditionalTestSet(
            root=Path(self.hparams.root), crop_size=self.hparams.crop_size, 
            nframes=self.hparams.nframes)
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, shuffle=False
        )