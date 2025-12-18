import os
import torch
from pathlib import Path
import argparse
from rich.progress import track
from PIL import Image

DEFAULT_ROOT = '/tmp_user/juno/sollivie/datasets/cloudy'
DEFAULT_NEW_ROOT = '/tmp_user/juno/sollivie/datasets/cloudy_frames'

def load_cloudy_image(path: Path):
    image = torch.load(path).squeeze()
    image = torch.flip(image, dims=(1,))
    image = image.transpose(0,1)
    return image

def partition_data(filepath: Path, newroot: Path):
    os.makedirs(newroot, exist_ok=True)
    data = load_cloudy_image(filepath)
    for idx in range(data.shape[0]):
        fp = newroot / f'{filepath.stem}_{idx:03d}.png'
        frame = data[idx]
        frame.squeeze_()
        ndarr = frame.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(ndarr, mode='L')
        im.save(fp)

def partition_root(root: Path, newroot: Path):
    for name in track(os.listdir(root), description='partition data'):
        filepath = root / name
        partition_data(filepath=filepath, newroot=newroot)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=DEFAULT_ROOT)
    parser.add_argument('--newroot', default=DEFAULT_NEW_ROOT)

    args = parser.parse_args()

    partition_root(Path(args.root), Path(args.newroot))

    
