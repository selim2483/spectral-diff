from math import ceil
import os
from typing import Sequence, Optional

import torch
from torchvision.utils import make_grid
import numpy as np
from matplotlib import cm, pyplot as plt
import wandb
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from texture_metrics.criteria.fourier import radial_profile
from metrics import _metric_dict, MetricsOptions

DEFAULT_IMAGES_TO_SHOW = ['samples', 'log_sp', 'diff_sp']
DEFAULT_METRICS_OPTIONS = MetricsOptions(
    metrics=[
        'mse', 
        'sliced_wasserstein_distance', 
        'spectral_radial_distance',
        'gradients_distance'
    ]
)

def to_batch_format(tensor: torch.Tensor, ndim=4):
    if tensor.ndim == ndim:
        return tensor
    else:
        return to_batch_format(tensor.unsqueeze(0), ndim)
    
def logsp(img: torch.Tensor):
    return torch.fft.fftshift(
        torch.fft.fft2(img.mean(dim=-3)).abs().log(), dim=(-1, -2))

def fourier_colormap(
        img: torch.Tensor, 
        vmin: Optional[float] = None, vmax: Optional[float] = None):
    vmin = vmin or img.min()
    vmax = vmax or img.max()
    img = (img - vmin) / (vmax - vmin)
    img = np.apply_along_axis(cm.viridis, 0, img.cpu().numpy())
    img = 2 * torch.from_numpy(np.squeeze(img)) - 1
    return img[..., :3, :, :]

def spectral_pool(x: torch.Tensor, nc_out: int = 3) -> torch.Tensor:
    """Perform mean pooling along spectral dimension.

    Args:
        x (torch.Tensor): image
        nc_out (int, optional): number of channels to output. 
            Defaults to 3.

    Returns:
        torch.Tensor: pooled image.
    """
    if x.ndim == 3:
        x.unsqueeze_(0)
    b, c, w, h = x.size()
    kernel_size = ceil(c / nc_out)
    if c==1 :
        return x.repeat((1, 3, 1, 1))
    else :
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        pooled = torch.nn.functional.avg_pool1d(
            x, kernel_size=kernel_size, ceil_mode=True)
        return pooled.permute(0, 2, 1).view(b, nc_out, w, h).squeeze()
    
def imlog(tensor: torch.Tensor, logger: str = 'wandb'):
    image = spectral_pool(tensor)
    if logger == 'wandb':
        return wandb.Image(image)
    else:
        image = image.clip(0, 1)
        return image

def make_grid_single_image(
        target: torch.Tensor, sample: torch.Tensor, 
        images_to_show: Sequence[str] = DEFAULT_IMAGES_TO_SHOW):
    print(target.shape, sample.shape)
    target.unsqueeze_(0)
    imgs = {}
    if 'samples' in images_to_show:
        imgs['samples'] = torch.cat([target, sample])
    if 'nn_patch' in images_to_show:
        pass
    if 'log_sp' in images_to_show:
        imgs['sp_t'] = logsp(target)
        imgs['log_sp'] = torch.cat([imgs['sp_t'], logsp(sample)])
        imgs['log_sp'] += imgs['log_sp'].min()
        if 'diff_sp' in images_to_show:
            imgs['diff_sp'] = (imgs['log_sp'] - imgs['sp_t']).abs()

    tensors = [
        imgs[key] 
        if key not in ['log_sp', 'diff_sp'] 
        else fourier_colormap(
            imgs[key], 
            vmin=0 if key=='log_sp' else None, 
            vmax=imgs['log_sp'].max() if key=='log_sp' else None) 
        for key in images_to_show
    ]

    for key in images_to_show:
        print(key, imgs[key].shape 
              if key not in ['log_sp', 'diff_sp'] 
              else fourier_colormap(
                  imgs[key], 
                  vmin=0 if key=='log_sp' else None, 
                  vmax=imgs['log_sp'].max() if key=='log_sp' else None).shape)

    grid = torch.cat(tensors)
    grid = make_grid(grid)
    print(grid.shape)
    grid = imlog(grid)

    return {'samples': grid}

def make_grid_multiple_images(
        target: torch.Tensor, sample: torch.Tensor, 
        images_to_show: Sequence[str] = DEFAULT_IMAGES_TO_SHOW):
    raise NotImplementedError

def make_grid_image(
        target: torch.Tensor, sample: torch.Tensor, 
        images_to_show: Sequence[str] = DEFAULT_IMAGES_TO_SHOW):
    target, sample = target.cpu(), sample.cpu()
    if target.ndim == 3 or target.size(0) == 1:
        return make_grid_single_image(target.squeeze(), sample, images_to_show)
    elif torch.all(target == target[0]):
        return make_grid_single_image(target[0], sample, images_to_show)
    else:
        return make_grid_multiple_images(target, sample, images_to_show)
    
def plot_radial_profile(target: torch.Tensor, sample: torch.Tensor, **kwargs):
    tensors = torch.cat([to_batch_format(target), to_batch_format(sample)])
    profiles = radial_profile(tensors.mean(dim=-3)).cpu()

    fig, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize', (10, 10)))
    for i, profile in enumerate(profiles):
        ax.plot(profile, label='target' if i==0 else f'sample {i}')

    ax.legend()
    ax.grid(True)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Fourier spectrum')
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()
    return fig

def log_metric_summary(
        pl_module: LightningModule, name: str, value: torch.Tensor):
    pl_module.log_dict({
        f'{name}/mean': value.mean(), 
        f'{name}/std': value.std(),
        f'{name}/min': value.min(),
        f'{name}/max': value.max()
    })  

class LogImagesSampleCallback(Callback):

    def __init__(
            self, images_to_show: Sequence[str] = DEFAULT_IMAGES_TO_SHOW, 
            radial_profile = True):
        super().__init__()
        self.images_to_show = images_to_show
        self.radial_profile = radial_profile

    def on_validation_batch_end(
            self, trainer: Trainer, pl_module: LightningModule, 
            outputs: dict, batch: dict, batch_idx: int, 
            dataloader_idx: int = 0):
        
        mu = batch.get('mean')
        target = batch.get('images') + mu
        sample = outputs.get('sample') + mu

        grids = make_grid_image(
            target, sample, images_to_show=self.images_to_show)
        trainer.logger.experiment.log(grids)

        if self.radial_profile:
            profile_plot = plot_radial_profile(target, sample)
            trainer.logger.experiment.log({'Radial Spectrum' : profile_plot})

class LogMetricsCallback(Callback):

    def __init__(self, options: MetricsOptions = DEFAULT_METRICS_OPTIONS):
        super().__init__()    
        self.options = options
        
    def on_validation_batch_end(
            self, trainer: Trainer, pl_module: LightningModule, 
            outputs: dict, batch: dict, batch_idx: int, 
            dataloader_idx: int = 0):
        
        mu = batch.get('mean')
        target = batch.get('images') + mu
        sample = outputs.get('sample') + mu

        for metric in self.options.metrics:
            value = _metric_dict[metric](target, sample, self.options)
            log_metric_summary(pl_module, metric, value)

def generate_unique_paths(logdir:str, logname:str):
    i = 0
    while(True):
        run_name = f'{logname}_{i:03}'
        if not os.path.isdir(os.path.join(logdir, run_name)) :
            return run_name
        i = i + 1
        

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(
            self, root = None, dirname = None, filename = None, monitor = None, 
            verbose = False, save_last = None, save_top_k = 1, 
            save_on_exception = False, save_weights_only = False, 
            mode = "min", auto_insert_metric_name = True, 
            every_n_train_steps = None, train_time_interval = None, 
            every_n_epochs = None, save_on_train_epoch_end = None, 
            enable_version_counter = True):
        
        super().__init__(
            dirpath, filename, monitor, verbose, save_last, save_top_k, 
            save_on_exception, save_weights_only, mode, 
            auto_insert_metric_name, every_n_train_steps, train_time_interval,
            every_n_epochs, save_on_train_epoch_end, enable_version_counter)