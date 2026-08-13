import torch
import torch.nn as nn
import math


class UNetBlock(nn.Module):
    """Базовый блок U-Net для DDPM"""
    def __init__(self, in_channels, out_channels, temb_dim=None, up=False, down=False):
        super().__init__()
        self.up = up
        self.down = down

        if temb_dim:
            self.time_emb_proj = nn.Linear(temb_dim, out_channels)

        if up:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        elif down:
            self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

    def forward(self, x, temb=None):
        x = self.conv(x)
        if temb is not None:
            temb = self.time_emb_proj(self.act(temb))[:, :, None, None]
            x = x + temb
        x = self.norm(x)
        return self.act(x)


class UNet(nn.Module):
    """
    U-Net архитектура для DDPM (Denoising Diffusion Probabilistic Models)
    Для изображений 128x128x3
    """
    def __init__(self, img_channels=3, base_channels=64, time_emb_dim=128):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, 1, 1)

        # Encoder (downsampling)
        self.enc1 = UNetBlock(base_channels, base_channels, time_emb_dim)
        self.enc2 = UNetBlock(base_channels, base_channels * 2, time_emb_dim, down=True)
        self.enc3 = UNetBlock(base_channels * 2, base_channels * 4, time_emb_dim, down=True)
        self.enc4 = UNetBlock(base_channels * 4, base_channels * 8, time_emb_dim, down=True)

        # Middle
        self.mid1 = UNetBlock(base_channels * 8, base_channels * 8, time_emb_dim)
        self.mid2 = UNetBlock(base_channels * 8, base_channels * 8, time_emb_dim)

        # Decoder (upsampling)
        self.dec4 = UNetBlock(base_channels * 16, base_channels * 8, time_emb_dim, up=True)
        self.dec3 = UNetBlock(base_channels * 12, base_channels * 4, time_emb_dim, up=True)
        self.dec2 = UNetBlock(base_channels * 6, base_channels * 2, time_emb_dim, up=True)
        self.dec1 = UNetBlock(base_channels * 3, base_channels, time_emb_dim, up=True)

        # Output
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(base_channels // 2, img_channels, 3, 1, 1),
        )

    def forward(self, x, t):
        # Time embedding
        t = t.view(-1, 1).float()
        t = self.time_mlp(t)

        # Initial conv
        x0 = self.init_conv(x)
        x0 = nn.SiLU()(x0)

        # Encoder
        e1 = self.enc1(x0, t)
        e2 = self.enc2(e1, t)
        e3 = self.enc3(e2, t)
        e4 = self.enc4(e3, t)

        # Middle
        m = self.mid1(e4, t)
        m = self.mid2(m, t)

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([m, e4], dim=1), t)
        d3 = self.dec3(torch.cat([d4, e3], dim=1), t)
        d2 = self.dec2(torch.cat([d3, e2], dim=1), t)
        d1 = self.dec1(torch.cat([d2, e1], dim=1), t)

        # Output
        return self.final_conv(d1)


def get_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02, schedule='linear'):
    """
    Расписание шума (beta schedule) для DDPM
    """
    if schedule == 'linear':
        betas = torch.linspace(beta_start, beta_end, timesteps)
    elif schedule == 'cosine':
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    return betas


def extract(a, t, x_shape):
    """Извлечение значений для текущего timestep"""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class DDPM:
    """
    DDPM (Denoising Diffusion Probabilistic Model) класс для обучения и генерации
    """
    def __init__(self, model, timesteps=1000, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.timesteps = timesteps

        # Beta schedule
        self.betas = get_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod = extract(torch.sqrt(self.alphas_cumprod), t, x_0.shape)
        sqrt_one_minus_alphas_cumprod = extract(torch.sqrt(1. - self.alphas_cumprod), t, x_0.shape)

        return sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise

    @torch.no_grad()
    def p_sample(self, x, t):
        """
        Reverse diffusion step: p(x_{t-1} | x_t)
        """
        model_output = self.model(x, t)

        # Coefficients
        sqrt_recip_alphas = torch.sqrt(1.0 / extract(self.alphas, t, x.shape))
        betas_t = extract(self.betas, t, x.shape)

        # Mean prediction
        if t[0] == 0:
            noise_pred = model_output
        else:
            noise_pred = model_output

        # Compute mean
        mean = sqrt_recip_alphas * (x - betas_t / torch.sqrt(1 - extract(self.alphas_cumprod, t, x.shape)) * noise_pred)

        # Variance
        variance = betas_t
        noise = torch.randn_like(x) if t[0] > 0 else 0

        return mean + torch.sqrt(variance) * noise

    @torch.no_grad()
    def sample(self, num_samples, img_shape=(3, 128, 128)):
        """
        Генерация новых изображений
        """
        x = torch.randn(num_samples, *img_shape, device=self.device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), i, dtype=torch.long, device=self.device)
            x = self.p_sample(x, t)

        return x.clamp(-1, 1)

    def loss_fn(self, x_0):
        """
        Функция потерь для обучения DDPM
        """
        batch_size = x_0.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)

        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        predicted_noise = self.model(x_t, t)

        return nn.MSELoss()(predicted_noise, noise)


if __name__ == '__main__':
    # Тестирование модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(img_channels=3, base_channels=64)
    ddpm = DDPM(model, timesteps=100, device=device)

    # Тест forward pass
    x = torch.randn(2, 3, 128, 128).to(device)
    t = torch.randint(0, 100, (2,)).to(device)
    output = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Подсчет параметров
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")