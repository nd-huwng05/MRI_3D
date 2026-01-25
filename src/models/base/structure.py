import torch
from timm.layers import trunc_normal_
from torch import nn
from .conv import ConvBlock, SinusoidalPositionEmbeddings, LayerNorm, ConvNeXtBlock


class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=4):
        super().__init__()
        #Downsampling: 256 -> 128 -> 64 -> 32
        self.net = nn.Sequential(
            ConvBlock(in_channels, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 256),
            nn.Conv2d(256, latent_dim * 2, 3, padding=1)  # Output: Mean & LogVar
        )

    def forward(self, x):
        return self.net(x)


class ConvNeXtEncoder(nn.Module):
    r"""
    ConvNeXt Encoder customized for VAE.
    Target: Compress 256x256 -> 32x32 (Downsample ratio = 8)
    """

    def __init__(self, in_chans=3, latent_dim=4,
                 depths=[3, 3, 3], dims=[64, 128, 256], drop_path_rate=0.,
                 layer_scale_init_value=1e-6):
        super().__init__()

        # Stem: Downsample 4x (256 -> 64)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        # Downsampling: (64 -> 32)
        for i in range(1):  # Chỉ loop 1 lần để downsample từ dims[0] sang dims[1]
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # Stage 0: Resolution 64x64, Dim 64
        # Stage 1: Resolution 32x32, Dim 128

        stage0 = nn.Sequential(
            *[ConvNeXtBlock(dim=dims[0], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in
              range(depths[0])])
        self.stages.append(stage0)
        cur += depths[0]

        stage1 = nn.Sequential(
            *[ConvNeXtBlock(dim=dims[1], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in
              range(depths[1])])
        self.stages.append(stage1)
        cur += depths[1]

        # Final Norm & Head (Projection to Latent)
        self.norm = LayerNorm(dims[1], eps=1e-6, data_format="channels_first")  # kernel 128

        # Output: Mean & LogVar (latent_dim * 2)
        self.head = nn.Conv2d(dims[1], latent_dim * 2, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.downsample_layers[0](x)  #64x64
        x = self.stages[0](x)

        x = self.downsample_layers[1](x)  #32x32
        x = self.stages[1](x)

        x = self.norm(x)
        x = self.head(x)  #(Batch, 8, 32, 32)
        return x

class Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=4):
        super().__init__()
        #Upsampling: 32 -> 64 -> 128 -> 256
        self.net = nn.Sequential(
            ConvBlock(latent_dim, 256),
            nn.Upsample(scale_factor=2),
            ConvBlock(256, 128),
            nn.Upsample(scale_factor=2),
            ConvBlock(128, 64),
            nn.Upsample(scale_factor=2),
            ConvBlock(64, 32),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.Tanh()  # Output range [-1, 1] for images
        )

    def forward(self, x):
        return self.net(x)

class SimpleVAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=4):
        super().__init__()
        self.encoder = ConvNeXtEncoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar, z

    def encode(self, x):
        h = self.encoder(x)
        mu, _ = torch.chunk(h, 2, dim=1)
        return mu


class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=4, time_dim=256):
        super().__init__()
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # Encoder (Down)
        self.down1 = ConvBlock(in_channels, 64)
        self.down2 = ConvBlock(64, 128)
        self.down3 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bot1 = ConvBlock(256, 512)

        self.time_proj = nn.Linear(time_dim, 512)

        self.bot2 = ConvBlock(512, 512)
        self.bot3 = ConvBlock(512, 256)

        # Decoder (Up)
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv_up1 = ConvBlock(256 + 256, 128)  # +256 do skip connection

        self.up2 = nn.Upsample(scale_factor=2)
        self.conv_up2 = ConvBlock(128 + 128, 64)

        self.up3 = nn.Upsample(scale_factor=2)
        self.conv_up3 = ConvBlock(64 + 64, 64)

        self.out = nn.Conv2d(64, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))

        x_bot = self.pool(x3)  # Shape: [Batch, 256, 4, 4]
        x_bot = self.bot1(x_bot)  # Shape: [Batch, 512, 4, 4]

        t_emb_proj = self.time_proj(t_emb)
        t_emb_proj = t_emb_proj[(...,) + (None,) * 2]

        x_bot = x_bot + t_emb_proj

        x_bot = self.bot2(x_bot)
        x_bot = self.bot3(x_bot)

        x_up1 = self.conv_up1(torch.cat([self.up1(x_bot), x3], dim=1))
        x_up2 = self.conv_up2(torch.cat([self.up2(x_up1), x2], dim=1))
        x_up3 = self.conv_up3(torch.cat([self.up3(x_up2), x1], dim=1))

        return self.out(x_up3)