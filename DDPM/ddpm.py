"""
DDPM для 128x128 RGB.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
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

        frequencies = torch.exp(
            torch.arange(
                half_dim,
                device=t.device,
                dtype=torch.float32,
            )
            * (
                -math.log(10000.0)
                / max(
                    half_dim - 1,
                    1,
                )
            )
        )

        values = (
            t.float()[:, None]
            * frequencies[None, :]
        )

        embedding = torch.cat(
            [
                values.sin(),
                values.cos(),
            ],
            dim=-1,
        )

        if self.dim % 2:
            embedding = F.pad(
                embedding,
                (0, 1),
            )

        return embedding


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        groups_1 = min(
            8,
            in_channels,
        )

        while (
            in_channels % groups_1 != 0
        ):
            groups_1 -= 1

        groups_2 = min(
            8,
            out_channels,
        )

        while (
            out_channels % groups_2 != 0
        ):
            groups_2 -= 1

        self.norm1 = nn.GroupNorm(
            groups_1,
            in_channels,
        )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
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
            groups_2,
            out_channels,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1,
            bias=False,
        )

        self.skip = (
            nn.Conv2d(
                in_channels,
                out_channels,
                1,
                bias=False,
            )
            if in_channels != out_channels
            else nn.Identity()
        )

        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv1(
            self.act(
                self.norm1(x)
            )
        )

        h = (
            h
            + self.time_proj(temb)[
                :, :, None, None
            ]
        )

        h = self.conv2(
            self.dropout(
                self.act(
                    self.norm2(h)
                )
            )
        )

        return h + self.skip(x)


class SelfAttention2d(nn.Module):
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
            1,
            bias=False,
        )

        self.proj = nn.Conv2d(
            channels,
            channels,
            1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = x.shape

        residual = x

        qkv = self.qkv(
            self.norm(x)
        )

        q, k, v = qkv.chunk(
            3,
            dim=1,
        )

        def split_heads(
            tensor: torch.Tensor,
        ) -> torch.Tensor:
            tensor = tensor.view(
                b,
                self.num_heads,
                self.head_dim,
                h * w,
            )

            return tensor.permute(
                0,
                1,
                3,
                2,
            )

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        attention = torch.softmax(
            torch.matmul(
                q,
                k.transpose(-2, -1),
            )
            * self.scale,
            dim=-1,
        )

        output = torch.matmul(
            attention,
            v,
        )

        output = (
            output.permute(
                0,
                1,
                3,
                2,
            )
            .contiguous()
            .view(
                b,
                c,
                h,
                w,
            )
        )

        return (
            self.proj(output)
            + residual
        )


class DownBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_emb_dim: int,
        use_attn: bool,
        dropout: float,
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
            4,
            2,
            1,
            bias=False,
        )

    def forward(
        self,
        x,
        temb,
    ):
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
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        time_emb_dim: int,
        use_attn: bool,
        dropout: float,
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_ch,
            in_ch,
            4,
            2,
            1,
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
        x,
        skip,
        temb,
    ):
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

        return self.attn(h)


class UNet(nn.Module):
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
            3,
            padding=1,
        )

        self.down1 = DownBlock(
            ch,
            ch * 2,
            time_emb_dim,
            False,
            dropout,
        )

        self.down2 = DownBlock(
            ch * 2,
            ch * 4,
            time_emb_dim,
            False,
            dropout,
        )

        # Attention начинает работать на 16x16.
        self.down3 = DownBlock(
            ch * 4,
            ch * 8,
            time_emb_dim,
            False,
            dropout,
        )

        self.down4 = DownBlock(
            ch * 8,
            ch * 8,
            time_emb_dim,
            True,
            dropout,
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

        self.up4 = UpBlock(
            ch * 8,
            ch * 8,
            ch * 8,
            time_emb_dim,
            True,
            dropout,
        )

        self.up3 = UpBlock(
            ch * 8,
            ch * 8,
            ch * 4,
            time_emb_dim,
            False,
            dropout,
        )

        self.up2 = UpBlock(
            ch * 4,
            ch * 4,
            ch * 2,
            time_emb_dim,
            False,
            dropout,
        )

        self.up1 = UpBlock(
            ch * 2,
            ch * 2,
            ch,
            time_emb_dim,
            False,
            dropout,
        )

        groups = min(8, ch)

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
                3,
                padding=1,
            ),
        )

    def forward(
        self,
        x,
        t,
    ):
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

        return self.out_conv(
            self.out_norm(x)
        )


def get_beta_schedule(
    timesteps: int,
    schedule: str = "cosine",
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
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

        x = torch.linspace(
            0,
            timesteps,
            timesteps + 1,
        )

        alpha_bar = torch.cos(
            (
                x / timesteps
                + s
            )
            / (1.0 + s)
            * math.pi
            * 0.5
        ).pow(2)

        alpha_bar /= alpha_bar[0]

        betas = 1.0 - (
            alpha_bar[1:]
            / alpha_bar[:-1]
        )

        return betas.clamp(
            1e-5,
            0.999,
        )

    raise ValueError(
        f"Unknown schedule: {schedule}"
    )


def extract(
    arr: torch.Tensor,
    t: torch.Tensor,
    shape,
) -> torch.Tensor:
    values = arr.to(
        t.device
    ).gather(
        0,
        t.long(),
    )

    return values.reshape(
        t.shape[0],
        *(
            1
            for _ in range(
                len(shape) - 1
            )
        ),
    )


class DDPM:
    """
    Важно:
    класс НЕ делает model.to(device).

    Модель сначала размещается на GPU,
    затем оборачивается DDP в train.py.
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        schedule: str = "cosine",
        device: str | torch.device = "cuda",
    ):
        self.model = model
        self.device = torch.device(
            device
        )
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

        self.sqrt_alphas_cumprod = torch.sqrt(
            self.alphas_cumprod
        )

        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0
            - self.alphas_cumprod
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
        ).clamp_min(1e-20)

        self.posterior_mean_coef1 = (
            betas
            * torch.sqrt(
                self.alphas_cumprod_prev
            )
            / (
                1.0
                - self.alphas_cumprod
            )
        )

        self.posterior_mean_coef2 = (
            (
                1.0
                - self.alphas_cumprod_prev
            )
            * torch.sqrt(self.alphas)
            / (
                1.0
                - self.alphas_cumprod
            )
        )

    def q_sample(
        self,
        x_0,
        t,
        noise: Optional[torch.Tensor] = None,
    ):
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

        x_t = (
            sqrt_alpha_bar * x_0
            + sqrt_one_minus * noise
        )

        return x_t, noise

    def loss_fn(
        self,
        x_0,
    ):
        batch_size = x_0.shape[0]

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
        x,
        t_idx: int,
    ):
        batch_size = x.shape[0]

        t = torch.full(
            (
                batch_size,
            ),
            t_idx,
            device=x.device,
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

        sqrt_one_minus = extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x.shape,
        )

        x0_pred = (
            x
            - sqrt_one_minus * noise_pred
        ) / torch.sqrt(alpha_bar)

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

        mean = (
            coef1 * x0_pred
            + coef2 * x
        )

        if t_idx == 0:
            return mean

        variance = extract(
            self.posterior_variance,
            t,
            x.shape,
        )

        return (
            mean
            + torch.sqrt(variance)
            * torch.randn_like(x)
        )

    @torch.no_grad()
    def sample(
        self,
        n: int,
        img_shape: tuple[int, int, int],
        verbose: bool = False,
    ):
        self.model.eval()

        x = torch.randn(
            n,
            *img_shape,
            device=self.device,
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
                    steps,
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
            -1,
            1,
        )

    @torch.no_grad()
    def ddim_sample(
        self,
        n: int,
        img_shape: tuple[int, int, int],
        sampling_steps: int = 50,
        eta: float = 0.0,
        verbose: bool = False,
    ):
        if sampling_steps <= 0:
            raise ValueError(
                "sampling_steps должен быть > 0."
            )

        sampling_steps = min(
            sampling_steps,
            self.timesteps,
        )

        self.model.eval()

        x = torch.randn(
            n,
            *img_shape,
            device=self.device,
        )

        time_indices = torch.linspace(
            0,
            self.timesteps - 1,
            sampling_steps,
            device=self.device,
        ).round().long()

        time_indices = torch.unique(
            time_indices
        ).flip(0)

        iterator = time_indices

        if verbose:
            try:
                from tqdm import tqdm

                iterator = tqdm(
                    time_indices,
                    total=len(time_indices),
                    desc="DDIM Sampling",
                )
            except ImportError:
                pass

        for index, current_t in enumerate(
            iterator
        ):
            current_t = int(
                current_t.item()
            )

            if index + 1 < len(time_indices):
                previous_t = int(
                    time_indices[
                        index + 1
                    ].item()
                )
            else:
                previous_t = -1

            t = torch.full(
                (n,),
                current_t,
                device=x.device,
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

            sqrt_alpha_bar = torch.sqrt(
                alpha_bar
            )

            sqrt_one_minus = torch.sqrt(
                1.0 - alpha_bar
            )

            x0_pred = (
                x
                - sqrt_one_minus * noise_pred
            ) / sqrt_alpha_bar

            x0_pred = x0_pred.clamp(
                -1,
                1,
            )

            if previous_t < 0:
                alpha_bar_prev = torch.ones_like(
                    alpha_bar
                )
            else:
                prev = torch.full(
                    (n,),
                    previous_t,
                    device=x.device,
                    dtype=torch.long,
                )

                alpha_bar_prev = extract(
                    self.alphas_cumprod,
                    prev,
                    x.shape,
                )

            sigma = (
                eta
                * torch.sqrt(
                    (
                        (
                            1.0
                            - alpha_bar_prev
                        )
                        / (
                            1.0
                            - alpha_bar
                        )
                    ).clamp_min(0.0)
                )
                * torch.sqrt(
                    (
                        1.0
                        - alpha_bar
                        / alpha_bar_prev
                    ).clamp_min(0.0)
                )
            )

            direction = torch.sqrt(
                (
                    1.0
                    - alpha_bar_prev
                    - sigma.pow(2)
                ).clamp_min(0.0)
            ) * noise_pred

            noise = (
                torch.randn_like(x)
                if eta > 0
                and previous_t >= 0
                else torch.zeros_like(x)
            )

            x = (
                torch.sqrt(
                    alpha_bar_prev
                )
                * x0_pred
                + direction
                + sigma * noise
            )

        return x.clamp(
            -1,
            1,
        )
