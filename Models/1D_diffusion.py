import sys
import torch
import math
import yaml

from torch import optim
from tqdm import trange
from models import DiffusionAttnUnet1D
from copy import deepcopy
from utils import ema_update
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.nn import functional as F
from torch.utils import data

from dataset import SampleDataset

# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def get_crash_schedule(t):
    sigma = torch.sin(t * math.pi / 2) ** 2
    alpha = (1 - sigma ** 2) ** 0.5
    return alpha_sigma_to_t(alpha, sigma)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2


@torch.no_grad()
def sample(model, x, steps, eta):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]

    t = get_crash_schedule(t)

    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * t[i]).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred


class DiffusionUncond(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()

        self.diffusion = DiffusionAttnUnet1D(configs["latent_dim"], io_channels=2, n_attn_layers=4)
        self.diffusion_ema = deepcopy(self.diffusion)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True, seed=configs["seed"])
        self.ema_decay = configs["ema_decay"]
        
    def configure_optimizers(self):
        return optim.Adam([*self.diffusion.parameters()], lr=4e-5)
  
    def training_step(self, batch, batch_idx):
        reals = batch[0]

        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        t = get_crash_schedule(t)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(reals)
        noised_reals = reals * alphas + noise * sigmas
        targets = noise * alphas - reals * sigmas

        with torch.cuda.amp.autocast():
            v = self.diffusion(noised_reals, t)
            mse_loss = F.mse_loss(v, targets)
            loss = mse_loss

        log_dict = {
            'train/loss': loss.detach(),
            'train/mse_loss': mse_loss.detach(),
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        decay = 0.95 if self.current_epoch < 25 else self.ema_decay
        ema_update(self.diffusion, self.diffusion_ema, decay)

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)


class DemoCallback(pl.Callback):
    def __init__(self, configs):
        super().__init__()
        self.demo_every = configs["demo_every"]
        self.num_demos = configs["num_demos"]
        self.demo_samples = configs["sample_size"]
        self.demo_steps = configs["demo_steps"]
        self.last_demo_step = -1

    @rank_zero_only
    @torch.no_grad()
    #def on_train_epoch_end(self, trainer, module):
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):        
  
        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return
        
        self.last_demo_step = trainer.global_step
    
        noise = torch.randn([self.num_demos, 2, self.demo_samples]).to(module.device)

        try:
            fakes = sample(module.diffusion_ema, noise, self.demo_steps, 0)
            print(fakes.shape)

            # Put the demos together
            # fakes = rearrange(fakes, 'b d n -> d (b n)')

            # log_dict = {}
            
            # filename = f'demo_{trainer.global_step:08}.wav'
            # fakes = fakes.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            # torchaudio.save(filename, fakes, self.sample_rate)


            # log_dict[f'demo'] = wandb.Audio(filename,
            #                                     sample_rate=self.sample_rate,
            #                                     caption=f'Demo')
        
            # log_dict[f'demo_melspec_left'] = wandb.Image(audio_spectrogram_image(fakes))

            # trainer.logger.experiment.log(log_dict, step=trainer.global_step)
        except Exception as e:
            print(f'{type(e).__name__}: {e}', file=sys.stderr)


def main():
    with open('config.yaml') as file:
        configs = yaml.safe_load(file)

    configs["latent_dim"] = 0
    save_path = configs["save_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device: ", device)
    torch.manual_seed(configs["seed"])

    train_set = SampleDataset(configs["training_dir"])
    train_dl = data.DataLoader(
        train_set, 
        configs["batch_size"], 
        shuffle=True, 
        num_workers=configs["num_workers"],
        persistent_workers=True,
        pin_memory=True
    )

    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=configs["checkpoint_every"], save_top_k=-1, dirpath=save_path)
    demo_callback = DemoCallback(configs)

    diffusion_model = DiffusionUncond(configs)

    diffusion_trainer = pl.Trainer(
        devices=configs["num_gpus"],
        accelerator="gpu",
        precision=16,
        accumulate_grad_batches=configs["accum_batches"],
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        max_epochs=100
    )

# , ckpt_path=configs["ckpt_path"]

    diffusion_trainer.fit(diffusion_model, train_dl)

if __name__ == '__main__':
    main()




