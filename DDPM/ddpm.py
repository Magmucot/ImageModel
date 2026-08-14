"""
DDPM (Denoising Diffusion Probabilistic Models) для изображений 128×128×3.

Улучшения по сравнению с базовой версией:
  - SinusoidalPositionEmbeddings — стандартное time embedding из оригинальной
    статьи DDPM (Ho et al., 2020), гораздо богаче чем простой nn.Linear(1, dim)
  - ResNetBlock — вместо простых ConvBlocks; поддерживает time embedding
  - SelfAttention2d — в bottleneck и на уровне 16×16, 8×8
  - Косинусное beta-расписание по умолчанию (лучше чем линейное для лиц)
  - p_sample возвращает x_{t-1} с правильным posterior variance
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Time Embedding
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal positional embeddings для временного шага t.
    Из оригинальной статьи DDPM (Ho et al., 2020) и Attention Is All You Need.

    PE(t, 2i)   = sin(t / 10000^(2i/d))
    PE(t, 2i+1) = cos(t / 10000^(2i/d))
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) — целые временные шаги
        Returns:
            (B, dim) — sinusoidal embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None].float() * embeddings[None, :]
        return torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Строительные блоки
# ─────────────────────────────────────────────────────────────────────────────

class ResNetBlock(nn.Module):
    """
    ResNet-блок с time embedding.

    Структура:
        x → Conv → GroupNorm → SiLU  ─────────────┐
                               + time_emb_proj     │ → + skip → SiLU
                              → GroupNorm → SiLU   │
                              → Conv → GroupNorm   │
                              → Dropout            ↑
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        time_emb_dim: int,
        dropout:      float = 0.1,
    ):
        super().__init__()
        groups = min(8, out_channels)

        self.norm1   = nn.GroupNorm(groups, in_channels)
        self.conv1   = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)

        # Проекция time embedding
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2   = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2   = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, C_in, H, W)
            temb: (B, time_emb_dim) — time embedding
        """
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Добавляем time embedding (broadcast по H, W)
        h = h + self.time_proj(temb)[:, :, None, None]

        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """
    Multi-head Self-Attention для 2D feature maps (как в DDPM статье).
    Применяется в bottleneck (8×8) и промежуточных уровнях.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.scale     = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv  = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        h = self.norm(x)

        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        def to_heads(t: torch.Tensor) -> torch.Tensor:
            t = t.view(B, self.num_heads, self.head_dim, H * W)
            return t.permute(0, 1, 3, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)
        attn = (torch.matmul(q, k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out  = torch.matmul(attn, v)
        out  = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)

        return self.proj(out) + residual


class DownBlock(nn.Module):
    """Энкодер-блок: 2 × ResNetBlock + опциональный Attention + Downsample."""

    def __init__(
        self,
        in_ch:        int,
        out_ch:       int,
        time_emb_dim: int,
        use_attn:     bool  = False,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.res1  = ResNetBlock(in_ch,  out_ch, time_emb_dim, dropout)
        self.res2  = ResNetBlock(out_ch, out_ch, time_emb_dim, dropout)
        self.attn  = SelfAttention2d(out_ch) if use_attn else nn.Identity()
        self.down  = nn.Conv2d(out_ch, out_ch, 4, 2, 1, bias=False)  # ↓2

    def forward(
        self, x: torch.Tensor, temb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns: (skip_features, downsampled_features)"""
        h = self.res1(x, temb)
        h = self.res2(h, temb)
        h = self.attn(h) if not isinstance(self.attn, nn.Identity) else h
        return h, self.down(h)


class UpBlock(nn.Module):
    """Декодер-блок: Upsample + конкатенация со skip + 2 × ResNetBlock + Attention."""

    def __init__(
        self,
        in_ch:        int,
        skip_ch:      int,
        out_ch:       int,
        time_emb_dim: int,
        use_attn:     bool  = False,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.up    = nn.ConvTranspose2d(in_ch, in_ch, 4, 2, 1, bias=False)  # ↑2
        self.res1  = ResNetBlock(in_ch + skip_ch, out_ch, time_emb_dim, dropout)
        self.res2  = ResNetBlock(out_ch, out_ch, time_emb_dim, dropout)
        self.attn  = SelfAttention2d(out_ch) if use_attn else nn.Identity()

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, temb: torch.Tensor
    ) -> torch.Tensor:
        h = self.up(x)
        h = torch.cat([h, skip], dim=1)
        h = self.res1(h, temb)
        h = self.res2(h, temb)
        return self.attn(h) if not isinstance(self.attn, nn.Identity) else h


# ─────────────────────────────────────────────────────────────────────────────
# UNet
# ─────────────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net для DDPM с Sinusoidal Time Embedding, ResNet-блоками и Self-Attention.
    Вход и выход: (B, 3, 128, 128).

    Уровни:
        128 × 128  (base_ch)
         64 × 64   (base_ch * 2)
         32 × 32   (base_ch * 4)
         16 × 16   (base_ch * 8)  ← Attention
          8 × 8    (base_ch * 8)  ← Attention (bottleneck)
    """

    def __init__(
        self,
        img_channels:  int   = 3,
        base_channels: int   = 64,
        time_emb_dim:  int   = 256,
        dropout:       float = 0.1,
    ):
        super().__init__()
        ch = base_channels

        # ── Time Embedding ─────────────────────────────────────────────────
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # ── Начальная свёртка ──────────────────────────────────────────────
        self.init_conv = nn.Conv2d(img_channels, ch, 3, 1, 1)

        # ── Encoder ───────────────────────────────────────────────────────
        # 128 → 64
        self.down1 = DownBlock(ch,     ch * 2,  time_emb_dim, use_attn=False, dropout=dropout)
        # 64  → 32
        self.down2 = DownBlock(ch * 2, ch * 4,  time_emb_dim, use_attn=False, dropout=dropout)
        # 32  → 16   (с attention)
        self.down3 = DownBlock(ch * 4, ch * 8,  time_emb_dim, use_attn=True,  dropout=dropout)
        # 16  → 8    (с attention)
        self.down4 = DownBlock(ch * 8, ch * 8,  time_emb_dim, use_attn=True,  dropout=dropout)

        # ── Bottleneck ─────────────────────────────────────────────────────
        self.mid_res1 = ResNetBlock(ch * 8, ch * 8, time_emb_dim, dropout)
        self.mid_attn = SelfAttention2d(ch * 8)
        self.mid_res2 = ResNetBlock(ch * 8, ch * 8, time_emb_dim, dropout)

        # ── Decoder ───────────────────────────────────────────────────────
        # 8  → 16   (с attention, skip от down4)
        self.up4 = UpBlock(ch * 8, ch * 8, ch * 8, time_emb_dim, use_attn=True,  dropout=dropout)
        # 16 → 32   (с attention, skip от down3)
        self.up3 = UpBlock(ch * 8, ch * 8, ch * 4, time_emb_dim, use_attn=True,  dropout=dropout)
        # 32 → 64   (skip от down2)
        self.up2 = UpBlock(ch * 4, ch * 4, ch * 2, time_emb_dim, use_attn=False, dropout=dropout)
        # 64 → 128  (skip от down1)
        self.up1 = UpBlock(ch * 2, ch * 2, ch,     time_emb_dim, use_attn=False, dropout=dropout)

        # ── Выходной блок ─────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(min(8, ch), ch)
        self.out_conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(ch, img_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 128, 128) — зашумлённое изображение x_t
            t: (B,)              — временные шаги (целые числа)
        Returns:
            (B, 3, 128, 128) — предсказанный шум ε_θ(x_t, t)
        """
        # Time embedding
        temb = self.time_mlp(t)          # (B, time_emb_dim)

        # Initial conv
        x = self.init_conv(x)            # (B, ch, 128, 128)

        # Encoder (сохраняем skip connections)
        skip1, x = self.down1(x, temb)   # skip1: (B, ch*2, 64, 64)
        skip2, x = self.down2(x, temb)   # skip2: (B, ch*4, 32, 32)
        skip3, x = self.down3(x, temb)   # skip3: (B, ch*8, 16, 16)
        skip4, x = self.down4(x, temb)   # skip4: (B, ch*8, 8,  8)

        # Bottleneck
        x = self.mid_res1(x, temb)
        x = self.mid_attn(x)
        x = self.mid_res2(x, temb)

        # Decoder
        x = self.up4(x, skip4, temb)     # (B, ch*8, 16, 16)
        x = self.up3(x, skip3, temb)     # (B, ch*4, 32, 32)
        x = self.up2(x, skip2, temb)     # (B, ch*2, 64, 64)
        x = self.up1(x, skip1, temb)     # (B, ch,  128, 128)

        return self.out_conv(self.out_norm(x))


# ─────────────────────────────────────────────────────────────────────────────
# Beta schedules
# ─────────────────────────────────────────────────────────────────────────────

def get_beta_schedule(
    timesteps:  int,
    schedule:   str   = "cosine",
    beta_start: float = 1e-4,
    beta_end:   float = 0.02,
) -> torch.Tensor:
    """
    Возвращает β_t расписание для T шагов.

    Args:
        timesteps:  T — количество шагов диффузии (обычно 1000)
        schedule:   'cosine' (рекомендуется для лиц) или 'linear'
        beta_start: начальное значение β (для linear)
        beta_end:   конечное значение β (для linear)
    """
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, timesteps)

    elif schedule == "cosine":
        # Nichol & Dhariwal, 2021 (Improved DDPM)
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alpha_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return betas.clamp(1e-4, 0.9999)

    else:
        raise ValueError(f"Неизвестное расписание: {schedule!r}")


def extract(
    arr: torch.Tensor,
    t:   torch.Tensor,
    shape: tuple,
) -> torch.Tensor:
    """
    Извлекает значения из arr по индексам t и reshapes для broadcast.

    arr:   (T,)
    t:     (B,) — индексы
    shape: (B, C, H, W) — целевая форма
    Returns: (B, 1, 1, 1) для broadcast с изображением
    """
    out = arr.gather(-1, t)
    return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))


# ─────────────────────────────────────────────────────────────────────────────
# DDPM класс
# ─────────────────────────────────────────────────────────────────────────────

class DDPM:
    """
    Обёртка над UNet, реализующая forward и reverse диффузию.

    Основные операции:
      q_sample  — forward process (добавление шума)
      loss_fn   — MSE(predicted_noise, real_noise)
      p_sample  — один шаг reverse process
      sample    — полная генерация (T → 0)
    """

    def __init__(
        self,
        model:     UNet,
        timesteps: int   = 1000,
        schedule:  str   = "cosine",
        device:    str   = "cuda",
    ):
        self.model     = model.to(device)
        self.device    = device
        self.timesteps = timesteps

        # Предвычисляем коэффициенты
        betas = get_beta_schedule(timesteps, schedule).to(device)

        self.betas             = betas
        self.alphas            = 1.0 - betas
        self.alphas_cumprod    = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Для q_sample
        self.sqrt_alphas_cumprod         = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Posterior variance для p_sample
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    # ── Forward process ───────────────────────────────────────────────────────

    def q_sample(
        self,
        x_0:   torch.Tensor,
        t:     torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Зашумляем x_0 до шага t: q(x_t | x_0) = N(√ᾱ_t · x_0, (1-ᾱ_t) · I)

        Returns:
            x_t:   зашумлённое изображение
            noise: добавленный шум (для вычисления loss)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = extract(self.sqrt_alphas_cumprod,          t, x_0.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus * noise
        return x_t, noise

    # ── Loss ─────────────────────────────────────────────────────────────────

    def loss_fn(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        Simplified DDPM loss: E[||ε - ε_θ(x_t, t)||²]

        Случайный t и случайный шум ε; модель предсказывает ε.
        """
        B  = x_0.size(0)
        t  = torch.randint(0, self.timesteps, (B,), device=self.device)
        x_t, noise = self.q_sample(x_0, t)
        noise_pred = self.model(x_t, t)
        return F.mse_loss(noise_pred, noise)

    # ── Reverse process ───────────────────────────────────────────────────────

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t_idx: int) -> torch.Tensor:
        """
        Один шаг обратного процесса: p_θ(x_{t-1} | x_t).

        Использует posterior mean + posterior variance (не просто β_t).
        """
        t_tensor = torch.full((x.shape[0],), t_idx, dtype=torch.long, device=self.device)

        # Предсказываем шум
        noise_pred = self.model(x, t_tensor)

        # Коэффициенты
        alpha      = extract(self.alphas,            t_tensor, x.shape)
        alpha_bar  = extract(self.alphas_cumprod,    t_tensor, x.shape)
        sqrt_recip = (1.0 / alpha).sqrt()
        betas_t    = extract(self.betas,             t_tensor, x.shape)
        sqrt_om_ab = extract(self.sqrt_one_minus_alphas_cumprod, t_tensor, x.shape)

        # Posterior mean
        mean = sqrt_recip * (x - betas_t / sqrt_om_ab * noise_pred)

        # Дисперсия (posterior variance)
        if t_idx == 0:
            return mean
        post_var = extract(self.posterior_variance, t_tensor, x.shape)
        noise = torch.randn_like(x)
        return mean + post_var.sqrt() * noise

    @torch.no_grad()
    def sample(
        self,
        n:         int   = 16,
        img_shape: tuple = (3, 128, 128),
        verbose:   bool  = False,
    ) -> torch.Tensor:
        """
        Полная генерация: xT ~ N(0,I) → x0.

        Args:
            n:         количество генерируемых изображений
            img_shape: (C, H, W)
            verbose:   выводить прогресс
        Returns:
            (n, C, H, W) — сгенерированные изображения в [-1, 1]
        """
        x = torch.randn(n, *img_shape, device=self.device)

        steps = list(reversed(range(self.timesteps)))
        if verbose:
            try:
                from tqdm import tqdm
                steps = tqdm(steps, desc="Sampling")
            except ImportError:
                pass

        for t_idx in steps:
            x = self.p_sample(x, t_idx)

        return x.clamp(-1, 1)




# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # UNet
    model = UNet(img_channels=3, base_channels=64, time_emb_dim=256).to(device)
    ddpm  = DDPM(model, timesteps=100, schedule="cosine", device=device)

    # Forward pass
    x = torch.randn(2, 3, 128, 128, device=device)
    t = torch.randint(0, 100, (2,), device=device)

    pred = model(x, t)
    print(f"Input:     {x.shape}")
    print(f"Pred:      {pred.shape}")

    # Loss
    loss = ddpm.loss_fn(x)
    print(f"Loss:      {loss.item():.4f}")

    # Параметры
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров: {n_params:,}  ({n_params / 1e6:.2f}M)")