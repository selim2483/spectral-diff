from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import torch

class SpectralDDPMScheduler(DDPMScheduler):

    def __init__(self, )
    def pure_noise(self, noise: torch.Tensor):
        return noise