"""
DDPM (Denoising Diffusion Probabilistic Models)
для изображений 128×128×3.

Особенности:

  - Sinusoidal time embedding.
  - ResNet blocks с time conditioning.
  - Self-attention только на 16×16 и 8×8.
  - Косинусное beta-расписание.
  - Корректный posterior variance.
  - Все timestep tensors создаются на device текущего batch.
  - Совместимость с single-GPU и DistributedDataParallel.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal positional embedding для timestep t.
    """

    def __init__(
        self,
        dim: int,
    ):
        super().__init__()

        if dim < 4:
            raise ValueError(
                "time_emb_dim должен быть >= 4."
            )

        self.dim = dim

    def forward(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        half_dim = self.dim // 2

        exponent = (
            math.log(10000)
            / (half_dim - 1)
        )

        embeddings = torch.exp(
            torch.arange(
                half_dim,
                device=t.device,
                dtype=torch.float32,
            )
            * -exponent
        )

        embeddings = (
            t[:, None].float()
            * embeddings[None, :]
        )

        embeddings = torch.cat(
            [
                embeddings.sin(),
                embeddings.cos(),
            ],
            dim=-1,
        )

        if self.dim % 2:
            embeddings = F.pad(
                embeddings,
                (0, 1),
            )

        return embeddings


class ResNetBlock(nn.Module):
    """
    ResNet block с time embedding.
    """

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
            out_channels,
        )

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
            groups,
            out_channels,
        )

        self.dropout = nn.Dropout(
            dropout
        )

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

        time = self.time_proj(
            temb
        )

        h = (
            h
            + time[:, :, None, None]
        )

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """
    Multi-head self-attention для feature map.

    Используется на небольших spatial resolution:
        16×16
        8×8
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
    ):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                "channels должен делиться "
                "на num_heads."
            )

        self.num_heads = num_heads

        self.head_dim = (
            channels // num_heads
        )

        self.scale = (
            self.head_dim ** -0.5
        )

        self.norm = nn.GroupNorm(
            min(8, channels),
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
        batch, channels, height, width = (
            x.shape
        )

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

        attention = (
            attention
            * self.scale
        )

        attention = attention.softmax(
            dim=-1
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

        return (
            self.proj(out)
            + residual
        )


class DownBlock(nn.Module):
    """
    Encoder block:

        ResNet
        ResNet
        optional attention
        downsample
    """

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

        self.attn = (
            SelfAttention2d(out_ch)
            if use_attn
            else nn.Identity()
        )

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
    """
    Decoder block:

        upsample
        concatenate skip
        ResNet
        ResNet
        optional attention
    """

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

        self.attn = (
            SelfAttention2d(out_ch)
            if use_attn
            else nn.Identity()
        )

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
            [
                h,
                skip,
            ],
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

        h = self.attn(h)

        return h


class UNet(nn.Module):
    """
    U-Net для DDPM.

    Для входа 128×128:

        128×128  base
         64×64   base*2
         32×32   base*4
         16×16   base*8 + attention
          8×8    base*8 + attention
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
            raise ValueError(
                "base_channels должен быть >= 8."
            )

        ch = base_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(
                time_emb_dim
            ),
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
        #
        # Attention здесь выключен,
        # потому что DownBlock применяет
        # attention до downsample, то есть
        # на resolution 32×32.
        self.down3 = DownBlock(
            ch * 4,
            ch * 8,
            time_emb_dim,
            use_attn=False,
            dropout=dropout,
        )

        # 16 -> 8
        #
        # Attention работает на 16×16.
        self.down4 = DownBlock(
            ch * 8,
            ch * 8,
            time_emb_dim,
            use_attn=True,
            dropout=dropout,
        )

        self.mid_res1 = ResNetBlock(
            ch * 8,
            ch * 8,
            time_emb_dim,
            dropout,
        )

        self.mid_attn = SelfAttention2d(
            ch * 8
        )

        self.mid_res2 = ResNetBlock(
            ch * 8,
            ch * 8,
            time_emb_dim,
            dropout,
        )

        # 8 -> 16
        #
        # Attention работает на 16×16.
        self.up4 = UpBlock(
            ch * 8,
            ch * 8,
            ch * 8,
            time_emb_dim,
            use_attn=True,
            dropout=dropout,
        )

        # 16 -> 32
        self.up3 = UpBlock(
            ch * 8,
            ch * 8,
            ch * 4,
            time_emb_dim,
            use_attn=False,
            dropout=dropout,
        )

        # 32 -> 64
        self.up2 = UpBlock(
            ch * 4,
            ch * 4,
            ch * 2,
            time_emb_dim,
            use_attn=False,
            dropout=dropout,
        )

        # 64 -> 128
        self.up1 = UpBlock(
            ch * 2,
            ch * 2,
            ch,
            time_emb_dim,
            use_attn=False,
            dropout=dropout,
        )

        self.out_norm = nn.GroupNorm(
            min(8, ch),
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


def get_beta_schedule(
    timesteps: int,
    schedule: str = "cosine",
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """
    Возвращает beta schedule.

    Args:
        timesteps:
            Количество diffusion steps.

        schedule:
            "cosine" или "linear".

        beta_start:
            Начальное beta для linear.

        beta_end:
            Конечное beta для linear.
    """

    if timesteps < 2:
        raise ValueError(
            "timesteps должен быть >= 2."
        )

    if schedule == "linear":
        return torch.linspace(
            beta_start,
            beta_end,
            timesteps,
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

        alpha_bar = (
            torch.cos(
                (
                    (x / timesteps)
                    + s
                )
                / (1 + s)
                * math.pi
                * 0.5
            )
            ** 2
        )

        alpha_bar = (
            alpha_bar
            / alpha_bar[0]
        )

        betas = 1.0 - (
            alpha_bar[1:]
            / alpha_bar[:-1]
        )

        return betas.clamp(
            min=1e-4,
            max=0.9999,
        )

    raise ValueError(
        f"Неизвестное расписание: "
        f"{schedule!r}"
    )


def extract(
    arr: torch.Tensor,
    t: torch.Tensor,
    shape: tuple,
) -> torch.Tensor:
    """
    Извлекает значения arr по индексам t.

    Все операции выполняются на device
    текущего batch.
    """

    if arr.device != t.device:
        arr = arr.to(t.device)

    out = arr.gather(
        0,
        t,
    )

    return out.reshape(
        t.shape[0],
        *((1,) * (len(shape) - 1)),
    )


class DDPM:
    """
    Обёртка над UNet, реализующая forward
    и reverse diffusion.

    ВАЖНО:

    DDPM не перемещает модель на device.

    Это позволяет безопасно использовать:

        UNet -> DDP -> DDPM

    вместо:

        UNet -> DDP -> DDPM.to(device)

    Device модели должен контролироваться
    train.py.
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        schedule: str = "cosine",
        device: str = "cuda",
    ):
        if timesteps < 2:
            raise ValueError(
                "timesteps должен быть >= 2."
            )

        self.model = model
        self.device = torch.device(
            device
        )
        self.timesteps = timesteps

        betas = get_beta_schedule(
            timesteps,
            schedule,
        ).to(
            self.device
        )

        self.betas = betas

        self.alphas = (
            1.0 - betas
        )

        self.alphas_cumprod = (
            torch.cumprod(
                self.alphas,
                dim=0,
            )
        )

        self.alphas_cumprod_prev = (
            F.pad(
                self.alphas_cumprod[:-1],
                (1, 0),
                value=1.0,
            )
        )

        self.sqrt_alphas_cumprod = (
            torch.sqrt(
                self.alphas_cumprod
            )
        )

        self.sqrt_one_minus_alphas_cumprod = (
            torch.sqrt(
                1.0
                - self.alphas_cumprod
            )
        )

        self.posterior_variance = (
            betas
            * (
                1.0
                - self.alphas_cumprod_prev
            )
            / (
                1.0
                - self.alphas_cumprod
            )
        )

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[
            torch.Tensor
        ] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Forward process:

            q(x_t | x_0)
            =
            N(
                sqrt(alpha_bar_t) * x_0,
                (1 - alpha_bar_t) * I
            )
        """

        if noise is None:
            noise = torch.randn_like(
                x_0
            )

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

        x_t = (
            sqrt_alpha_bar * x_0
            + sqrt_one_minus * noise
        )

        return x_t, noise

    def loss_fn(
        self,
        x_0: torch.Tensor,
    ) -> torch.Tensor:
        """
        DDPM noise prediction loss.
        """

        batch_size = x_0.size(0)

        t = torch.randint(
            0,
            self.timesteps,
            (
                batch_size,
            ),
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

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t_idx: int,
    ) -> torch.Tensor:
        """
        Один шаг обратного процесса:

            p_theta(x_{t-1} | x_t)
        """

        t_tensor = torch.full(
            (
                x.shape[0],
            ),
            t_idx,
            dtype=torch.long,
            device=x.device,
        )

        noise_pred = self.model(
            x,
            t_tensor,
        )

        alpha = extract(
            self.alphas,
            t_tensor,
            x.shape,
        )

        betas_t = extract(
            self.betas,
            t_tensor,
            x.shape,
        )

        sqrt_one_minus_alpha_bar = (
            extract(
                self.sqrt_one_minus_alphas_cumprod,
                t_tensor,
                x.shape,
            )
        )

        sqrt_recip_alpha = (
            torch.rsqrt(alpha)
        )

        mean = (
            sqrt_recip_alpha
            * (
                x
                - (
                    betas_t
                    / sqrt_one_minus_alpha_bar
                )
                * noise_pred
            )
        )

        if t_idx == 0:
            return mean

        posterior_variance = extract(
            self.posterior_variance,
            t_tensor,
            x.shape,
        )

        noise = torch.randn_like(
            x
        )

        return (
            mean
            + posterior_variance.sqrt()
            * noise
        )

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
    ) -> torch.Tensor:
        """
        Полная генерация:

            x_T ~ N(0, I)
            x_T -> ... -> x_0
        """

        x = torch.randn(
            n,
            *img_shape,
            device=self.device,
        )

        steps = reversed(
            range(
                self.timesteps
            )
        )

        if verbose:
            try:
                from tqdm import tqdm

                steps = tqdm(
                    steps,
                    desc="Sampling",
                )
            except ImportError:
                pass

        for t_idx in steps:
            x = self.p_sample(
                x,
                t_idx,
            )

        return x.clamp(
            -1,
            1,
        )


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    model = UNet(
        img_channels=3,
        base_channels=64,
        time_emb_dim=256,
    ).to(device)

    ddpm = DDPM(
        model=model,
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
        (
            2,
        ),
        device=device,
    )

    pred = model(
        x,
        t,
    )

    print(
        f"Input: {x.shape}"
    )

    print(
        f"Pred:  {pred.shape}"
    )

    loss = ddpm.loss_fn(
        x
    )

    print(
        f"Loss:  {loss.item():.4f}"
    )

    n_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Параметров: "
        f"{n_params:,} "
        f"({n_params / 1e6:.2f}M)"
    )