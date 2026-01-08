import h5py
import numpy as np
import os
from rich.progress import track
import torch

base_path = os.getcwd().split('/sollivie/')[0]
data_path = os.path.join(base_path, 'sollivie', 'datasets', 'cloudy')
save_path = os.path.join(base_path, 'sollivie', 'datasets', 'cloud_0.hdf5')

hf = h5py.File(save_path, 'a')
cube_path = os.path.join(data_path, 'cloud_0.pt')
grp = hf.create_group(cube_path)

with open(cube_path, 'rb') as f:
    binary_data = f.read()

data = np.asarray(binary_data)
dset = hf.create_dataset('cloud_0', data=data)
hf.close()

print(f'torch file size: {os.path.getsize(cube_path) / (1024**2):.2f} MB')
print(f'hdf5 file size: {os.path.getsize(save_path) / (1024**2):.2f} MB')

# for i in os.listdir(base_path): 
#     cube_path = os.path.join(base_path, i)

# for j in track(os.listdir(cube_name)):
#     frame_path = os.path.join(cube_name, j)
    
#     with open(frame_path, 'rb') as f:
#         binary_data = f.read()
    
#     binary_array = np.asarray(binary_data)
#     dset = grp.create_dataset(j, data=binary_array)
