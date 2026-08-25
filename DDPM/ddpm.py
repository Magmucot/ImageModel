"""
DDPM (Denoising Diffusion Probabilistic Models)
для изображений 128x128x3.

Особенности:

- Sinusoidal time embedding.
- ResNet blocks с time conditioning.
- Self-attention на 16x16 и 8x8.
- Косинусное beta-расписание.
- Корректный posterior variance.
- DDPM sampling.
- DDIM sampling.
- Совместимость с Single-GPU.
- Совместимость с DistributedDataParallel.
- Все timestep tensors создаются на device текущего batch.
- Коэффициенты diffusion заранее вычисляются.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Time embedding
# ============================================================================


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embedding для diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()

        if dim < 4:
            raise ValueError("time_emb_dim должен быть >= 4.")

        self.dim = dim

    def forward(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        half_dim = self.dim // 2

        exponent = math.log(10000.0) / max(half_dim - 1, 1)

        frequencies = torch.exp(
            torch.arange(
                half_dim,
                device=t.device,
                dtype=torch.float32,
            )
            * -exponent
        )

        embeddings = t.float()[:, None] * frequencies[None, :]

        embeddings = torch.cat(
            (
                embeddings.sin(),
                embeddings.cos(),
            ),
            dim=-1,
        )

        if self.dim % 2:
            embeddings = F.pad(
                embeddings,
                (0, 1),
            )

        return embeddings


# ============================================================================
# ResNet block
# ============================================================================


class ResNetBlock(nn.Module):
    """ResNet block с conditioning по timestep."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        groups = min(
            8,
            in_channels,
        )

        while in_channels % groups != 0:
            groups -= 1

        out_groups = min(
            8,
            out_channels,
        )

        while out_channels % out_groups != 0:
            out_groups -= 1

        self.norm1 = nn.GroupNorm(
            groups,
            in_channels,
        )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                time_emb_dim,
                out_channels,
            ),
        )

        self.norm2 = nn.GroupNorm(
            out_groups,
            out_channels,
        )

        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        if in_channels != out_channels:
            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            )
        else:
            self.skip = nn.Identity()

        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        time_emb = self.time_proj(temb)

        h = h + time_emb[:, :, None, None]

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


# ============================================================================
# Self attention
# ============================================================================


class SelfAttention2d(nn.Module):
    """Multi-head self-attention для 2D feature map."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
    ):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError("channels должен делиться на num_heads.")

        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5

        groups = min(
            8,
            channels,
        )

        while channels % groups != 0:
            groups -= 1

        self.norm = nn.GroupNorm(
            groups,
            channels,
        )

        self.qkv = nn.Conv2d(
            channels,
            channels * 3,
            kernel_size=1,
            bias=False,
        )

        self.proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = x.shape

        residual = x

        h = self.norm(x)

        qkv = self.qkv(h)

        q, k, v = qkv.chunk(
            3,
            dim=1,
        )

        def to_heads(
            tensor: torch.Tensor,
        ) -> torch.Tensor:
            tensor = tensor.view(
                batch,
                self.num_heads,
                self.head_dim,
                height * width,
            )

            return tensor.permute(
                0,
                1,
                3,
                2,
            )

        q = to_heads(q)
        k = to_heads(k)
        v = to_heads(v)

        attention = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        attention = attention * self.scale

        attention = torch.softmax(
            attention,
            dim=-1,
        )

        out = torch.matmul(
            attention,
            v,
        )

        out = (
            out.permute(
                0,
                1,
                3,
                2,
            )
            .contiguous()
            .view(
                batch,
                channels,
                height,
                width,
            )
        )

        return self.proj(out) + residual


# ============================================================================
# Encoder / decoder blocks
# ============================================================================


class DownBlock(nn.Module):
    """Encoder block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_emb_dim: int,
        use_attn: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.res1 = ResNetBlock(
            in_ch,
            out_ch,
            time_emb_dim,
            dropout,
        )

        self.res2 = ResNetBlock(
            out_ch,
            out_ch,
            time_emb_dim,
            dropout,
        )

        self.attn = SelfAttention2d(out_ch) if use_attn else nn.Identity()

        self.down = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        h = self.res1(
            x,
            temb,
        )

        h = self.res2(
            h,
            temb,
        )

        h = self.attn(h)

        return h, self.down(h)


class UpBlock(nn.Module):
    """Decoder block."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        time_emb_dim: int,
        use_attn: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_ch,
            in_ch,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )

        self.res1 = ResNetBlock(
            in_ch + skip_ch,
            out_ch,
            time_emb_dim,
            dropout,
        )

        self.res2 = ResNetBlock(
            out_ch,
            out_ch,
            time_emb_dim,
            dropout,
        )

        self.attn = SelfAttention2d(out_ch) if use_attn else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        h = self.up(x)

        if h.shape[-2:] != skip.shape[-2:]:
            h = F.interpolate(
                h,
                size=skip.shape[-2:],
                mode="nearest",
            )

        h = torch.cat(
            (
                h,
                skip,
            ),
            dim=1,
        )

        h = self.res1(
            h,
            temb,
        )

        h = self.res2(
            h,
            temb,
        )

        return self.attn(h)


# ============================================================================
# UNet
# ============================================================================


class UNet(nn.Module):
    """
    U-Net для DDPM.

    Для изображения 128x128:

        128x128 -> base
         64x64  -> base*2
         32x32  -> base*4
         16x16  -> base*8 + attention
          8x8   -> base*8 + attention
    """

    def __init__(
        self,
        img_channels: int = 3,
        base_channels: int = 64,
        time_emb_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        if base_channels < 8:
            raise ValueError("base_channels должен быть >= 8.")

        ch = base_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(
                time_emb_dim,
                time_emb_dim * 4,
            ),
            nn.GELU(),
            nn.Linear(
                time_emb_dim * 4,
                time_emb_dim,
            ),
        )

        self.init_conv = nn.Conv2d(
            img_channels,
            ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # 128 -> 64
        self.down1 = DownBlock(
            ch,
            ch * 2,
            time_emb_dim,
            use_attn=False,
            dropout=dropout,
        )

        # 64 -> 32
        self.down2 = DownBlock(
            ch * 2,
            ch * 4,
            time_emb_dim,
            use_attn=False,
            dropout=dropout,
        )

        # 32 -> 16
        self.down3 = DownBlock(
            ch * 4,
            ch * 8,
            time_emb_dim,
            use_attn=False,
            dropout=dropout,
        )

        # 16 -> 8.
        # Attention выполняется до downsample,
        # поэтому resolution здесь = 16x16.
        self.down4 = DownBlock(
            ch * 8,
            ch * 8,
            time_emb_dim,
            use_attn=True,
            dropout=dropout,
        )

        # Bottleneck: 8x8.
        self.mid_res1 = ResNetBlock(
            ch * 8,
            ch * 8,
            time_emb_dim,
            dropout,
        )

        self.mid_attn = SelfAttention2d(ch * 8)

        self.mid_res2 = ResNetBlock(
            ch * 8,
            ch * 8,
            time_emb_dim,
            dropout,
        )

        # 8 -> 16.
        self.up4 = UpBlock(
            ch * 8,
            ch * 8,
            ch * 8,
            time_emb_dim,
            use_attn=True,
            dropout=dropout,
        )

        # 16 -> 32.
        self.up3 = UpBlock(
            ch * 8,
            ch * 8,
            ch * 4,
            time_emb_dim,
            use_attn=False,
            dropout=dropout,
        )

        # 32 -> 64.
        self.up2 = UpBlock(
            ch * 4,
            ch * 4,
            ch * 2,
            time_emb_dim,
            use_attn=False,
            dropout=dropout,
        )

        # 64 -> 128.
        self.up1 = UpBlock(
            ch * 2,
            ch * 2,
            ch,
            time_emb_dim,
            use_attn=False,
            dropout=dropout,
        )

        groups = min(
            8,
            ch,
        )

        while ch % groups != 0:
            groups -= 1

        self.out_norm = nn.GroupNorm(
            groups,
            ch,
        )

        self.out_conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(
                ch,
                img_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        temb = self.time_mlp(t)

        x = self.init_conv(x)

        skip1, x = self.down1(
            x,
            temb,
        )

        skip2, x = self.down2(
            x,
            temb,
        )

        skip3, x = self.down3(
            x,
            temb,
        )

        skip4, x = self.down4(
            x,
            temb,
        )

        x = self.mid_res1(
            x,
            temb,
        )

        x = self.mid_attn(x)

        x = self.mid_res2(
            x,
            temb,
        )

        x = self.up4(
            x,
            skip4,
            temb,
        )

        x = self.up3(
            x,
            skip3,
            temb,
        )

        x = self.up2(
            x,
            skip2,
            temb,
        )

        x = self.up1(
            x,
            skip1,
            temb,
        )

        x = self.out_norm(x)

        return self.out_conv(x)


# ============================================================================
# Beta schedule
# ============================================================================


def get_beta_schedule(
    timesteps: int,
    schedule: str = "cosine",
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Создаёт beta schedule."""

    if timesteps <= 0:
        raise ValueError("timesteps должен быть > 0.")

    if schedule == "linear":
        return torch.linspace(
            beta_start,
            beta_end,
            timesteps,
            dtype=torch.float32,
        )

    if schedule == "cosine":
        s = 0.008

        steps = timesteps + 1

        x = torch.linspace(
            0,
            timesteps,
            steps,
            dtype=torch.float32,
        )

        alpha_bar = torch.cos((x / timesteps + s) / (1.0 + s) * math.pi * 0.5).pow(2)

        alpha_bar = alpha_bar / alpha_bar[0]

        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])

        return betas.clamp(
            1e-5,
            0.999,
        )

    raise ValueError(f"Неизвестное расписание: {schedule!r}")


# ============================================================================
# Tensor helpers
# ============================================================================


def extract(
    arr: torch.Tensor,
    t: torch.Tensor,
    shape: tuple | torch.Size,
) -> torch.Tensor:
    """
    Извлекает arr[t] и приводит к форме:

        (B, 1, 1, 1)

    для изображения.
    """

    if t.dtype != torch.long:
        t = t.long()

    arr = arr.to(
        device=t.device,
    )

    out = arr.gather(
        0,
        t,
    )

    return out.reshape(
        t.shape[0],
        *(1 for _ in range(len(shape) - 1)),
    )


# ============================================================================
# DDPM
# ============================================================================


class DDPM:
    """
    DDPM wrapper.

    Поддерживает:

    - q_sample()
    - loss_fn()
    - p_sample()
    - sample()
    - ddim_sample()
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        schedule: str = "cosine",
        device: str | torch.device = "cuda",
    ):
        self.device = torch.device(device)

        self.model = model.to(self.device)

        self.timesteps = timesteps
        self.schedule = schedule

        betas = get_beta_schedule(
            timesteps,
            schedule,
        ).to(self.device)

        self.betas = betas

        self.alphas = 1.0 - betas

        self.alphas_cumprod = torch.cumprod(
            self.alphas,
            dim=0,
        )

        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1],
            (1, 0),
            value=1.0,
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)

        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_variance = self.posterior_variance.clamp(min=1e-20)

        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    # ========================================================================
    # Forward diffusion
    # ========================================================================

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        q(x_t | x_0).

        x_0 -> x_t.
        """

        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = extract(
            self.sqrt_alphas_cumprod,
            t,
            x_0.shape,
        )

        sqrt_one_minus = extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x_0.shape,
        )

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus * noise

        return x_t, noise

    # ========================================================================
    # Training loss
    # ========================================================================

    def loss_fn(
        self,
        x_0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard epsilon-prediction DDPM loss.
        """

        batch_size = x_0.shape[0]

        t = torch.randint(
            0,
            self.timesteps,
            (batch_size,),
            device=x_0.device,
            dtype=torch.long,
        )

        x_t, noise = self.q_sample(
            x_0,
            t,
        )

        noise_pred = self.model(
            x_t,
            t,
        )

        return F.mse_loss(
            noise_pred,
            noise,
        )

    # ========================================================================
    # DDPM reverse process
    # ========================================================================

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t_idx: int,
    ) -> torch.Tensor:
        """
        Один шаг:

            x_t -> x_{t-1}

        через DDPM posterior.
        """

        batch_size = x.shape[0]

        t = torch.full(
            (batch_size,),
            t_idx,
            device=x.device,
            dtype=torch.long,
        )

        noise_pred = self.model(
            x,
            t,
        )

        alpha = extract(
            self.alphas,
            t,
            x.shape,
        )

        alpha_bar = extract(
            self.alphas_cumprod,
            t,
            x.shape,
        )

        beta = extract(
            self.betas,
            t,
            x.shape,
        )

        sqrt_one_minus_alpha_bar = extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x.shape,
        )

        # Predicted x0.
        x0_pred = (x - sqrt_one_minus_alpha_bar * noise_pred) / torch.sqrt(alpha_bar)

        x0_pred = x0_pred.clamp(
            -1.0,
            1.0,
        )

        coef1 = extract(
            self.posterior_mean_coef1,
            t,
            x.shape,
        )

        coef2 = extract(
            self.posterior_mean_coef2,
            t,
            x.shape,
        )

        posterior_mean = coef1 * x0_pred + coef2 * x

        if t_idx == 0:
            return posterior_mean

        variance = extract(
            self.posterior_variance,
            t,
            x.shape,
        )

        noise = torch.randn_like(x)

        return posterior_mean + torch.sqrt(variance) * noise

    # ========================================================================
    # DDPM sampling
    # ========================================================================

    @torch.no_grad()
    def sample(
        self,
        n: int = 16,
        img_shape: tuple = (
            3,
            128,
            128,
        ),
        verbose: bool = False,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """
        Полная DDPM генерация.

        Возвращает tensor в [-1, 1].
        """

        sample_device = torch.device(device) if device is not None else self.device

        self.model.eval()

        x = torch.randn(
            n,
            *img_shape,
            device=sample_device,
        )

        steps = range(
            self.timesteps - 1,
            -1,
            -1,
        )

        if verbose:
            try:
                from tqdm import tqdm

                steps = tqdm(
                    list(steps),
                    total=self.timesteps,
                    desc="DDPM Sampling",
                )
            except ImportError:
                pass

        for t_idx in steps:
            x = self.p_sample(
                x,
                int(t_idx),
            )

        return x.clamp(
            -1.0,
            1.0,
        )

    # ========================================================================
    # DDIM sampling
    # ========================================================================

    @torch.no_grad()
    def ddim_sample(
        self,
        n: int = 16,
        img_shape: tuple = (
            3,
            128,
            128,
        ),
        sampling_steps: int = 50,
        eta: float = 0.0,
        verbose: bool = False,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """
        DDIM sampling.

        sampling_steps=50 значительно быстрее
        полного DDPM sampling с T=1000.

        eta=0:
            deterministic DDIM.

        eta=1:
            stochastic DDIM.
        """

        if sampling_steps <= 0:
            raise ValueError("sampling_steps должен быть > 0.")

        if sampling_steps > self.timesteps:
            raise ValueError("sampling_steps не может быть больше timesteps.")

        sample_device = torch.device(device) if device is not None else self.device

        self.model.eval()

        x = torch.randn(
            n,
            *img_shape,
            device=sample_device,
        )

        timesteps = (
            torch.linspace(
                0,
                self.timesteps - 1,
                sampling_steps,
                device=sample_device,
            )
            .round()
            .long()
        )

        timesteps = torch.unique(timesteps)

        timesteps = timesteps.flip(0)

        if verbose:
            try:
                from tqdm import tqdm

                timesteps = tqdm(
                    timesteps.tolist(),
                    desc="DDIM Sampling",
                )
            except ImportError:
                pass

        for index, t_idx_tensor in enumerate(timesteps):
            t_idx = int(t_idx_tensor)

            t = torch.full(
                (n,),
                t_idx,
                device=sample_device,
                dtype=torch.long,
            )

            noise_pred = self.model(
                x,
                t,
            )

            alpha_bar = extract(
                self.alphas_cumprod,
                t,
                x.shape,
            )

            sqrt_alpha_bar = torch.sqrt(alpha_bar)

            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

            x0_pred = (x - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar

            x0_pred = x0_pred.clamp(
                -1.0,
                1.0,
            )

            if index == len(timesteps) - 1:
                alpha_bar_prev = torch.ones_like(alpha_bar)
            else:
                prev_t = int(timesteps[index + 1])

                prev_t_tensor = torch.full(
                    (n,),
                    prev_t,
                    device=sample_device,
                    dtype=torch.long,
                )

                alpha_bar_prev = extract(
                    self.alphas_cumprod,
                    prev_t_tensor,
                    x.shape,
                )

            sigma = (
                eta
                * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
                * torch.sqrt(1.0 - alpha_bar / alpha_bar_prev)
            )

            direction = (
                torch.sqrt((1.0 - alpha_bar_prev - sigma.pow(2)).clamp(min=0.0))
                * noise_pred
            )

            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

            x = torch.sqrt(alpha_bar_prev) * x0_pred + direction + sigma * noise

        return x.clamp(
            -1.0,
            1.0,
        )


# ============================================================================
# Smoke test
# ============================================================================


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    model = UNet(
        img_channels=3,
        base_channels=32,
        time_emb_dim=128,
        dropout=0.1,
    ).to(device)

    ddpm = DDPM(
        model,
        timesteps=100,
        schedule="cosine",
        device=device,
    )

    x = torch.randn(
        2,
        3,
        128,
        128,
        device=device,
    )

    t = torch.randint(
        0,
        100,
        (2,),
        device=device,
    )

    pred = model(
        x,
        t,
    )

    print(f"Input: {tuple(x.shape)}")

    print(f"Pred:  {tuple(pred.shape)}")

    loss = ddpm.loss_fn(x)

    print(f"Loss: {loss.item():.4f}")

    n_params = sum(p.numel() for p in model.parameters())

    print(f"Параметров: {n_params:,} ({n_params / 1e6:.2f}M)")
