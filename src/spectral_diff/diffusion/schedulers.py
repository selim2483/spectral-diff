from typing import Callable, List, Optional, Union
from diffusers.schedulers.scheduling_utils import SchedulerMixin
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput
# from diffusers.schedulers.scheduling_ddim import DDIMScheduler

import diffusers
from diffusers.configuration_utils import ConfigMixin, register_to_config
import numpy as np
import torch
    
class DDPMScheduler(diffusers.schedulers.scheduling_ddpm.DDPMScheduler):

    def noise_function(self, noise: torch.Tensor):
        return noise
    
class DDIMScheduler(diffusers.schedulers.scheduling_ddim.DDIMScheduler):

    def noise_function(self, noise: torch.Tensor):
        return noise