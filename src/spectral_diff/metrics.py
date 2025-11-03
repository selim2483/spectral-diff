from dataclasses import dataclass, field
from typing import Callable

from texture_metrics.criteria import fourier
from texture_metrics.criteria import gradients
from texture_metrics.criteria import optimal_transport
from texture_metrics.criteria import style_distances
from texture_metrics.criteria import CNN, CNNOptions, RandomTripletDataset
from texture_metrics.transforms import get_stats
import torch

_metric_dict = dict()

def is_valid_metric(metric):
    return metric in _metric_dict

def list_valid_metrics():
    return list(_metric_dict.keys())

def register_metric(func: Callable):
    assert callable(func)
    _metric_dict[func.__name__] = func
    
    return func

# -------------------------------------------------------------------------- #

@dataclass
class MetricsOptions:
    """Texture synthesis metrics options"""
    # Random seed for reproductibility.
    seed:         int  = 0
    # Overwrites the existing metrics.yaml file.
    overwrite:    bool = True
    # Metrics to compute. Full list available in :metrics.py:.
    metrics:      list = field(default_factory=list)
    # Transformation to perform on images befor computing metrics.
    # Follows the same typing as the :transforms: parameter in 
    # :synthesis.py:
    # Use raw: none to perform no transformation.
    transforms:   dict = field(default_factory=dict)
    # Style distance batch size.
    bstyle:       int  = 1
    # Number of slice for SWD on CNN features.
    sstyle:       int  = 1
    # Neural statistics to use : 'mean', 'gram', 'covariance', 'swd'.
    fstyle:       list = field(
        default_factory=lambda: ['mean', 'gram', 'covariance', 'swd'])
    # Projection to use for the :style_distance_projected: metric.
    projections:  dict = field(default_factory=dict)
    # Number of slices for SWD.
    nhist:        int  = 1000
    # SWD batch size.
    bhist:        int  = 250
    # Number of slices for SWD on gradients images.
    ngrad:        int  = 1000
    # Gradients SWD batch size.
    bgrad:        int  = 250
    
    # Model Options to use as a CNN for deep features extraction and style
    # distances computation
    cnn:          CNNOptions = field(default_factory=CNNOptions)

@register_metric
def mse(target: torch.Tensor, synth: torch.Tensor, options: MetricsOptions):
    return torch.mean((synth - target)**2, dim=(-1,-2,-3))

@register_metric
def sliced_wasserstein_distance(
        target: torch.Tensor, synth: torch.Tensor, options: MetricsOptions):
    """Computes Sliced Wasserstein Distance (SWD) between target and
    synthetic images.

    Args:
        target (torch.Tensor): target image.
        synth (torch.Tensor): synthetic image.
        options (MetricsOptions): metrics options.

    Returns:
        Number: SWD
    """
    return optimal_transport.sliced_wasserstein_distance(
        target, synth, nslice=options.nhist, batch_size=options.bhist)

@register_metric
def spectral_radial_distance(
        target: torch.Tensor, synth: torch.Tensor, options: MetricsOptions):
    """Computes L-2 distance on azimuthal spectra (mean and band-wise).

    Args:
        target (torch.Tensor): target image.
        synth (torch.Tensor): synthetic image.
        options (MetricsOptions): metrics options.

    Returns:
        dict: dictionnary containing mean and band-wise radial
            spectral distances.
    """
    return fourier.spectral_radial_distance(
        target.mean(dim=-3), synth.mean(dim=-3)).sqrt()

@register_metric
def gradients_distance(
        target: torch.Tensor, synth: torch.Tensor, options: MetricsOptions):
    """Computes gradients distribution distances (along x and y axis
    and magnitude).

    Args:
        target (torch.Tensor): target image.
        synth (torch.Tensor): synthetic image.
        options (MetricsOptions): metrics options.

    Returns:
        dict: dictionnary containing gradients distances.
    """
    return optimal_transport.sliced_wasserstein_distance(
        gradients.image_gradient(target, result='mag'),
        gradients.image_gradient(synth, result='mag'),
        nslice=options.ngrad, batch_size=options.bgrad
    )