from dataclasses import dataclass, field
from math import ceil
import os
import random
from typing import Callable, Sequence, Optional

import torch
from torchvision.utils import make_grid
import numpy as np
from matplotlib import cm, pyplot as plt
import wandb
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from texture_metrics.criteria.fourier import radial_profile
from metrics import _metric_dict, MetricsOptions

@dataclass
class LoggingOptions:
    loggings: list = field(default_factory=list)
    image_viz: Sequence[str] = field(
        default_factory=lambda: DEFAULT_IMAGES_TO_SHOW)
    plt_kwargs: Sequence[int] = field(default_factory=dict)
    # video_viz: Sequence[str] = field(
    #     default_factory=lambda: DEFAULT_IMAGES_TO_SHOW)

DEFAULT_IMAGE_LOGGING_OPTIONS = LoggingOptions(
    loggings=['make_grid_image', 'plot_radiale_profile'])
DEFAULT_VIDEO_LOGGING_OPTIONS = LoggingOptions(
    loggings=['random_slice'])
DEFAULT_IMAGES_TO_SHOW = ['samples', 'log_sp', 'diff_sp']
DEFAULT_METRICS_OPTIONS = MetricsOptions(
    metrics=[
        'mse', 
        'sliced_wasserstein_distance', 
        'spectral_radial_distance',
        'gradients_distance'
    ]
)
DEFAULT_VIDEO_VIZ = ['random_slice', 'z_profile', 'animation']

# -------------------------------------------------------------------------- #

_logging_dict = dict()

def is_valid_logging(logging):
    return logging in _logging_dict

def list_valid_loggings():
    return list(_logging_dict.keys())

def register_logging(func: Callable):
    assert callable(func)
    _logging_dict[func.__name__] = func
    
    return func
# --------------------------------- Image ---------------------------------- #

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
    target.unsqueeze_(0)
    imgs = {}
    if 'samples' in images_to_show:
        imgs['samples'] = torch.cat([target, sample])
    if 'nn_patch' in images_to_show:
        pass
    if 'log_sp' in images_to_show:
        imgs['sp_t'] = logsp(target)
        imgs['log_sp'] = torch.cat([imgs['sp_t'], logsp(sample)])
        # imgs['log_sp'] += imgs['log_sp'].min()
        if 'diff_sp' in images_to_show:
            imgs['diff_sp'] = (imgs['log_sp'] - imgs['sp_t']).abs()

    tensors = [
        imgs[key] 
        if key not in ['log_sp', 'diff_sp'] 
        else fourier_colormap(
            imgs[key], 
            vmin=0 if key=='diff_sp' else None, 
            vmax=imgs['log_sp'].max() if key=='diff_sp' else None) 
        for key in images_to_show
    ]

    grid = torch.cat(tensors)
    grid = make_grid(
        grid, nrow=tensors[0].shape[0], normalize=True, value_range=(-1,1))
    grid = imlog(grid)

    return {'samples': grid}

def make_grid_multiple_images(
        target: torch.Tensor, sample: torch.Tensor, 
        images_to_show: Sequence[str] = DEFAULT_IMAGES_TO_SHOW):
    raise NotImplementedError

@register_logging
def make_grid_image(
        target: torch.Tensor, sample: torch.Tensor, options: LoggingOptions):
    target, sample = target.cpu(), sample.cpu()
    if target.ndim == 3 or target.size(0) == 1:
        return make_grid_single_image(
            target.squeeze(), sample, options.image_viz)
    elif torch.all(target == target[0]):
        return make_grid_single_image(target[0], sample, options.image_viz)
    else:
        return make_grid_multiple_images(target, sample, options.image_viz)

@register_logging
def plot_radial_profile(
        target: torch.Tensor, sample: torch.Tensor, options: LoggingOptions):
    tensors = torch.cat([to_batch_format(target), to_batch_format(sample)])
    profiles = radial_profile(tensors.mean(dim=-3)).cpu()

    fig, ax = plt.subplots(
        1, 1, figsize=options.plt_kwargs.get('figsize', (10, 10)))
    for i, profile in enumerate(profiles):
        ax.plot(profile, label='target' if i==0 else f'sample {i}')

    ax.legend()
    ax.grid(True)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Fourier spectrum')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Fourier Spectrum Radial Profile')

    plt.tight_layout()
    return {'Radial Spectrum' : fig}

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
            self, options: LoggingOptions = DEFAULT_IMAGE_LOGGING_OPTIONS):
        super().__init__()
        self.options = options

    def on_validation_batch_end(
            self, trainer: Trainer, pl_module: LightningModule, 
            outputs: dict, batch: dict, batch_idx: int, 
            dataloader_idx: int = 0):
        
        mu = batch.get('mean')
        target = batch.get('images') + mu
        sample = outputs.get('sample') + mu

        for logging in self.options.loggings:
            output = _logging_dict[logging](target, sample, self.options)
            trainer.logger.experiment.log(output, step=trainer.global_step)

# --------------------------------- Video ---------------------------------- #

@register_logging
def log_random_frame(
        target: torch.Tensor, sample: torch.Tensor, options: LoggingOptions):
    idx = random.randint(1,target.shape[1])
    grid = make_grid_image(target[:,idx,:,:], sample[:,idx,:,:], options)
    return {'random slice samples': grid[:,idx,:,:]}

@register_logging
def plot_vertical_profile(
        target: torch.Tensor, sample: torch.Tensor, options: LoggingOptions):
    target_profiles = target.mean(dim=(-1,-2)).cpu()
    sample_profiles = sample.mean(dim=(-1,-2)).cpu()

    colors = plt.cm.tab10
    fig, ax = plt.subplots(
        1, 1, figsize=options.plt_kwargs.get('figsize', (10, 10)))
    for i in enumerate(target_profiles.shape[0]):
        c = colors(i % 10)
        ax.plot(
            target_profiles[i], label=f'target {i}', linestyle='-', color=c)
        ax.plot(
            sample_profiles[i], label=f'sample {i}', linestyle='-', color=c)

    ax.legend()
    ax.grid(True)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel('Water content')
    ax.set_title('Vertical profile')
    plt.tight_layout()

    return fig

@register_logging
def make_animation(
        target: torch.Tensor, sample: torch.Tensor, options: LoggingOptions):
    grid = make_grid(
        image, nrow=image[0].shape[0], normalize=True, value_range=(-1,1))
    grid.squeeze_().unsqueeze_(1)

class LogVideoSampleCallback(Callback):

    def __init__(self, video_viz: Sequence[str] = DEFAULT_VIDEO_VIZ):
        super().__init__()

    def on_validation_batch_end(
            self, trainer: Trainer, pl_module: LightningModule, 
            outputs: dict, batch: dict, batch_idx: int, 
            dataloader_idx: int = 0):
        
        target = batch.get('images')
        sample = outputs.get('sample')

# -------------------------------- Metrics --------------------------------- #

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
