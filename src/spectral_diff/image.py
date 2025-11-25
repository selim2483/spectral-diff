from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.models.unets import UNet2DModel
from jsonargparse import lazy_instance

from data import AugmentedDataModule, SingleImageDataModule
from diffusion import Diffusion, ConditionalDiffusion 
from utils.logging import LogImagesSampleCallback, LogMetricsCallback

# DEFAULT_MODEL = {'class_path': 'diffusers.models.unets.UNet2DModel'}
# DEFAULT_SCHEDULER = {'class_path': 'diffusers.schedulers.scheduling_ddpm.DDPMScheduler'}
# DEFAULT_MODULE = Diffusion(
#     model=UNet2DModel(sample_size=256), scheduler=DDPMScheduler())
# DEFAULT_DATA = SingleImageDataModule(
#     image_path='/scratchm/sollivie/datasets/DeepTextures/tissu.png')
# DEFAULT_LOGGER = WandbLogger(name='test', project='spectralDiff')
# DEFAULT_CALLBACKS = [
#     ModelSummary(max_depth=3), 
#     ModelCheckpoint(
#         dirpath='/tmp_user/juno/sollivie/spectral-diff/runs/test',
#         filename='model-{step}', save_last=True, every_n_train_steps=10_000,
#         save_top_k=-1),
#     LogMetricsCallback(), 
#     LogImagesSampleCallback(),
# ]

class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.set_defaults({
            # 'model': lazy_instance(
            #     Diffusion, 
            #     model=UNet2DModel(sample_size=256),
            #     scheduler=DDPMScheduler()),
            'model.model': lazy_instance(UNet2DModel, sample_size=256), 
            'model.scheduler': lazy_instance(DDPMScheduler),
            'data': lazy_instance(
                SingleImageDataModule, 
                image_path='/tmp_user/juno/sollivie/datasets/DeepTextures/tissu.png'),
            'trainer.max_steps': 50_000,
            'trainer.val_check_interval': 5_000,
            'trainer.check_val_every_n_epoch': None,
            'trainer.logger': lazy_instance(
                WandbLogger, name='test', project='spectralDiff'),
            # 'trainer.callbacks': [
            #     lazy_instance(ModelSummary, max_depth=3), 
            #     lazy_instance(
            #         ModelCheckpoint, 
            #         dirpath='/tmp_user/juno/sollivie/spectral-diff/runs/test',
            #         filename='model-{step}', save_last=True, 
            #         every_n_train_steps=10_000, save_top_k=-1),
            #     lazy_instance(LogMetricsCallback), 
            #     lazy_instance(LogImagesSampleCallback)],
        })
        # parser.add_argument('exp_name', default='test')
        # parser.add_argument('exp_name_unique')
        # parser.link_arguments(('trainer.callbacks', 'exp_name'), 'exp_name_unique', compute_fn=lambda name: )
        # parser.link_arguments('exp_name', 'trainer.logger.name')
        parser.link_arguments(
            'trainer.val_check_interval', 'data.init_args.sample_every_n_steps')
        # parser.link_arguments('')
        # parser.link_arguments('trainer.logger.save_dir', 'trainer.callbacks.')
        
def cli_main():
    cli = CustomCLI(
        Diffusion, save_config_kwargs={"overwrite": True})

if __name__=="__main__":
    cli_main()