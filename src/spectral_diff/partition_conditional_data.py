import torch
from torchvision.utils import save_image
from pathlib import Path

def load_cloudy_image(path: Path):
    image = torch.load(path).squeeze()
    image = torch.flip(image, dims=(1,))
    image = image.transpose(0,1)
    return image

def partition_data(filepath: Path, newroot: Path):
    data = load_cloudy_image(filepath)
    for idx in range(data.shape[0]):
        framepath = newroot / f'{filepath.name}_{idx}'
        save_image(data[idx], framepath, normalize=True, value_range=(0.,1.))
