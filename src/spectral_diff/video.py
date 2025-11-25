from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.models.unets import UNet2DModel
from jsonargparse import lazy_instance

from data import CloudyDataModule
from diffusion import ConditionalDiffusion
from models.unet import UNet 

class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.set_defaults({
            'model.model': lazy_instance(UNet, dim=32, out_dim=3, channels=6), 
            'model.scheduler': lazy_instance(DDPMScheduler),
            'data': lazy_instance(
                CloudyDataModule, 
                root='/tmp_user/juno/sollivie/datasets/cloudy',
                nframes=10, batch_size=1, num_workers=1),
            'trainer.max_steps': 50_000,
            'trainer.val_check_interval': 5_000,
            'trainer.check_val_every_n_epoch': None,
            'trainer.logger': lazy_instance(
                WandbLogger, name='test', project='spectralDiff'),
        })
        parser.link_arguments(
            'trainer.val_check_interval', 'data.init_args.sample_every_n_steps')
        
def cli_main():
    cli = CustomCLI(
        ConditionalDiffusion, save_config_kwargs={"overwrite": True})

if __name__=="__main__":
    cli_main()