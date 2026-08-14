"""Обучение DDPM на 2x T4 (DDP, AMP FP16, In-Memory Dataset)."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import get_dataloaders
from DDPM.models import UNet
from utils.distributed import (
    cleanup_distributed,
    is_distributed,
    is_main_process,
    reduce_tensor,
    setup_distributed,
    wrap_ddp,
)
from utils.utils import load_config, print_model_info, save_checkpoint, save_samples


class GaussianDiffusion(nn.Module):
    """Модуль расписания диффузии с сохранением тензоров в буферах устройства."""

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(
            -1, 1, 1, 1
        )
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise


def train_epoch(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images in train_loader:
        images = images.to(device, non_blocking=True)
        batch_size = images.size(0)
        optimizer.zero_grad(set_to_none=True)

        t = torch.randint(
            0, diffusion.timesteps, (batch_size,), device=device, dtype=torch.long
        )
        noise = torch.randn_like(images)
        x_noisy = diffusion.q_sample(images, t, noise=noise)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            predicted_noise = model(x_noisy, t)
            loss = nn.functional.mse_loss(predicted_noise, noise)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += reduce_tensor(loss.detach()).item()

    return total_loss / len(train_loader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ddpm.yaml")
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device, local_rank = setup_distributed()

    train_loader, _, train_sampler, _ = get_dataloaders(
        data_root=args.data_root,
        batch_size=cfg.get("batch_size", 32),
        num_workers=cfg.get("num_workers", 2),
        val_frac=0.0,
        distributed=is_distributed(),
        in_memory=True,
        image_size=cfg.get("image_size", 128),
    )

    diffusion = GaussianDiffusion(timesteps=cfg.get("timesteps", 1000)).to(device)
    model = UNet().to(device)
    print_model_info(model, "DDPM UNet")

    model = wrap_ddp(model, local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 0.0002))
    scaler = torch.amp.GradScaler("cuda")

    epochs = cfg.get("epochs", 100)
    save_dir = Path(cfg.get("save_dir", "checkpoints/ddpm"))

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_epoch(
            model,
            diffusion,
            train_loader,
            optimizer,
            scaler,
            device,
        )

        if is_main_process():
            print(f"[Epoch {epoch:03d}/{epochs:03d}] Loss: {train_loss:.5f}")

            if epoch % cfg.get("save_interval", 10) == 0 or epoch == epochs:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model": model,
                        "optimizer": optimizer,
                        "scaler": scaler,
                        "config": cfg,
                    },
                    is_best=False,
                    checkpoint_dir=save_dir,
                    filename=f"ddpm_epoch_{epoch:03d}.pt",
                )

    cleanup_distributed()


if __name__ == "__main__":
    main()
