from dataclasses import dataclass
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
import torch

from models.modules import Attention, ConvNextBlock, Downsample, LinearAttention, PreNorm, Residual, SinusoidalPosEmb, Upsample, default, exists
from utils.resize_right import resize


@dataclass
class UNet2DOutput(BaseOutput):
    """
    The output of [`UNet2DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.Tensor
    
class UNet(ModelMixin, ConfigMixin):
    
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["norm"]

    @register_to_config
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        with_time_emb = True
    ):
        super().__init__()

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = torch.nn.Sequential(
                SinusoidalPosEmb(dim),
                torch.nn.Linear(dim, dim * 4),
                torch.nn.GELU(),
                torch.nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(torch.nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, emb_dim=time_dim, norm =ind != 0),
                ConvNextBlock(dim_out, dim_out, emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else torch.nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(torch.nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, emb_dim=time_dim),
                ConvNextBlock(dim_in, dim_in, emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else torch.nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = torch.nn.Sequential(
            ConvNextBlock(dim, dim),
            torch.nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x: torch.Tensor, t: int):
        t_tensor = torch.full(
            (x.size(0),), t, dtype=torch.int64, device=x.device)
        t = self.time_mlp(t_tensor) if exists(self.time_mlp) else None
        initial_shape = x.shape[-2:]
        h = []

        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for convnext, convnext2, attn, upsample in self.ups:
            res = h.pop()
            x = torch.cat((resize(x, out_shape=res.shape[-2:]), res), dim=1)
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(resize(x, out_shape=initial_shape))