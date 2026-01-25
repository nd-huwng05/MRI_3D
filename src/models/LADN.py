import torch
from torch import nn
from .base.structure import SimpleVAE, DiffusionUNet


class LatentDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        # VAE: Compresses 3 channel image -> 4 channel latent
        self.vae = SimpleVAE(in_channels=3, latent_dim=4)

        # U-Net: Denoises 4 channel latent
        self.unet = DiffusionUNet(in_channels=4)

if __name__ == "__main__":
    print("Testing Architecture...")
    x = torch.randn(2, 3, 256, 256)
    t = torch.randint(0, 1000, (2,))

    model = LatentDiffusion()
    recon, mu, logvar, latents = model.vae(x)
    print(f"VAE Input: {x.shape} -> Latent: {latents.shape}")

    noise = model.unet(latents, t)
    print(f"Diffusion Output: {noise.shape}")