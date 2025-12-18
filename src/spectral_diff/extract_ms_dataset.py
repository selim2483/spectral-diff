import os
import torch
import numpy as np
import tqdm
import rasterio

root = '/scratchm/sollivie/datasets/hytexila/unzip'
torch_root = '/scratchm/sollivie/datasets/hytexila/multispectral'
# torch_root = '/scratchm/sollivie/datasets/hytexila/ms'
bands = [5,18,32,45,55,65,95,122,149,176]
os.makedirs(torch_root, exist_ok=True)
for name in tqdm.tqdm(os.listdir(root)):
    torch_path = os.path.join(torch_root, f'{name}.pt')
    if not os.path.exists(torch_path):
        path = os.path.join(root, name, f'{name}.raw')
        with rasterio.open(path) as src:
            data = src.read()
            data = torch.from_numpy(data)
            data = data[bands,:,:]
        torch.save(data, torch_path)