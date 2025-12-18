import torch
from diffusion import Diffusion

class ConditionalDiffusion(Diffusion):

    def set_example_input_array(self):
        self.example_input_array = torch.Tensor(8, 1, 256, 256)

    def training_step(self, batch, batch_idx):
        x0 = batch["image"]
        x0_cond = batch["condition"]
        noise = torch.randn_like(x0)
        t = torch.randint(
            self.scheduler.config.num_train_timesteps, 
            (x0.size(0),), device=self.device)
        x_noisy = self.scheduler.add_noise(x0, noise, t)
        output = self.model(torch.cat([x0_cond, x_noisy], dim=1), t).sample

        if self.scheduler.config.prediction_type == 'sample':
            loss = torch.nn.functional.mse_loss(output, x0)
        elif self.scheduler.config.prediction_type == 'epsilon':
            loss = torch.nn.functional.mse_loss(output, noise)
        else:
            raise NotImplementedError(
                "'prediction' argument should be 'x0' or 'epsilon'")
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def conditional_sample(
            self, condition: torch.Tensor, custom_timesteps: int = None):
        noise = torch.randn_like(condition)
        img = self.scheduler.noise_function(noise)
        condition = condition.to(self.device)

        self.scheduler.set_timesteps(
            custom_timesteps or self.num_inference_steps)
        for t in self.scheduler.timesteps:
            input = torch.cat([img, condition], dim=1)
            output = self.ema_model(input, t).sample
            img = self.scheduler.step(output, t, img).prev_sample

        return img
    
    def forward(
            self, start_frame: torch.Tensor, nframes: int = 10, 
            custom_timesteps: int = None):
        samples = [start_frame]
        for _ in range(1, nframes):
            next_frame = self.conditional_sample(
                samples[-1], custom_timesteps=custom_timesteps)
            samples.append(next_frame)

        return torch.cat(samples, dim=1)
        
    def validation_step(self, batch, batch_idx):
        img = batch.get('images')
        start_frame = img[:,0,:,:].unsqueeze(1)
        sample = self.forward(start_frame, img.shape[1])

        return {
            'loss': torch.nn.functional.mse_loss(img, sample),
            'sample': sample
        }

        
