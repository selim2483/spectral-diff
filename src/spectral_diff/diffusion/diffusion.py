import torch
import diffusers
import lightning as L

from models.modules import EMA
    
class Diffusion(L.LightningModule):

    def __init__(
            self, model: diffusers.models.ModelMixin, 
            scheduler: diffusers.schedulers.SchedulerMixin, 
            ema_beta: float = 0.999, initial_lr: float = 2e-5, 
            prediction: str = 'x0', num_inference_steps: int = None
        ):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.ema_beta = ema_beta
        self.initial_lr = initial_lr
        self.prediction = prediction.lower()
        self.num_inference_steps = num_inference_steps or self.scheduler.config.num_train_timesteps

        self.ema_model = EMA(self.model, self.ema_beta)
        self.example_input_array = torch.Tensor(8, 3, 256, 256)

    def calc_loss(
            self, x0: torch.Tensor, noise: torch.Tensor, output: torch.Tensor):
        if self.prediction == 'x0':
            return torch.nn.functional.mse_loss(x0, output)
        elif self.prediction == 'epsilon':
            return torch.nn.functional.mse_loss(noise, output)
        else:
            raise NotImplementedError(
                "'prediction' argument should be 'x0' or 'epsilon'")
    
    def training_step(self, batch, batch_idx):
        x0 = batch["images"]
        noise = torch.randn_like(x0)
        t = torch.randint(
            self.scheduler.config.num_train_timesteps, 
            (x0.size(0),), device=self.device)
        x_noisy = self.scheduler.add_noise(x0, noise, t)
        output = self.model(x_noisy, t).sample
        loss = self.calc_loss(x0, noise, output)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def forward(self, noise: torch.Tensor, custom_timesteps: int = None):
        img = noise.to(self.device)
        self.scheduler.set_timesteps(
            custom_timesteps or self.num_inference_steps)
        for t in self.scheduler.timesteps:
            print(t)
            output = self.ema_model(img, t).sample
            img = self.scheduler.step(output, t, img).prev_sample

        return img

    def on_after_backward(self):
        if self.ema_model:
            self.ema_model.update(self.model)

    def validation_step(self, batch, batch_idx):
        x = batch.get('images')

        noise = torch.randn_like(x)
        if hasattr(self.scheduler, 'pure_noise') and callable(self.scheduler.pure_noise):
            x_noise = self.scheduler.pure_noise(noise)
        else:
            x_noise = noise
        sample = self.forward(x_noise)

        return {
            'loss': torch.nn.functional.mse_loss(x, sample),
            'sample': sample
        }