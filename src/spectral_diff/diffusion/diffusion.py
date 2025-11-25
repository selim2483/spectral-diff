import torch
import diffusers
import lightning as L

from models.modules import EMA
    
class Diffusion(L.LightningModule):

    def __init__(
            self, model: diffusers.models.ModelMixin, 
            scheduler: diffusers.schedulers.SchedulerMixin, 
            ema_beta: float = 0.999, initial_lr: float = 2e-5, 
            num_inference_steps: int = None
        ):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.ema_beta = ema_beta
        self.initial_lr = initial_lr
        self.num_inference_steps = num_inference_steps \
            or self.scheduler.config.num_train_timesteps

        self.ema_model = EMA(self.model, self.ema_beta)
        self.save_hyperparameters()
        self.set_example_input_array()

    def set_example_input_array(self):
        self.example_input_array = torch.Tensor(8, 3, 256, 256)
    
    def training_step(self, batch, batch_idx):
        x0 = batch["images"]
        noise = torch.randn_like(x0)
        t = torch.randint(
            self.scheduler.config.num_train_timesteps, 
            (x0.size(0),), device=self.device)
        x_noisy = self.scheduler.add_noise(x0, noise, t)
        output = self.model(x_noisy, t).sample

        if self.scheduler.config.prediction_type == 'sample':
            loss = torch.nn.functional.mse_loss(output, x0)
        elif self.scheduler.config.prediction_type == 'epsilon':
            loss = torch.nn.functional.mse_loss(output, noise)
        else:
            raise NotImplementedError(
                "'prediction' argument should be 'sample' or 'epsilon'")
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[20], gamma=0.1)
        return [optimizer], [scheduler]

    def forward(self, x_noise: torch.Tensor, custom_timesteps: int = None):
        img = x_noise.to(self.device)
        self.scheduler.set_timesteps(
            custom_timesteps or self.num_inference_steps)
        
        for t in self.scheduler.timesteps:
            output = self.ema_model(img, t).sample
            img = self.scheduler.step(output, t, img).prev_sample

        return img

    def on_after_backward(self):
        if self.ema_model:
            self.ema_model.update(self.model)
        
    def validation_step(self, batch, batch_idx):
        x = batch.get('images')

        noise = torch.randn_like(x)
        x_noise = self.scheduler.noise_function(noise)
        sample = self.forward(x_noise)

        return {
            'loss': torch.nn.functional.mse_loss(x, sample),
            'x_noise': x_noise,
            'sample': sample
        }