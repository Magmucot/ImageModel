"""Обучение VAE на 2x T4 (DDP, AMP FP16, In-Memory Dataset)."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import get_dataloaders
from utils.distributed import (
    cleanup_distributed,
    is_distributed,
    is_main_process,
    reduce_tensor,
    setup_distributed,
    wrap_ddp,
)
from utils.utils import load_config, print_model_info, save_checkpoint, save_samples
from VAE.models import VAE


def vae_loss_fn(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kld_weight: float = 0.00025,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total_loss = recon_loss + kld_weight * kld_loss
    return total_loss, recon_loss, kld_loss


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    kld_weight: float,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images in train_loader:
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            recon, mu, logvar = model(images)
            loss, _, _ = vae_loss_fn(recon, images, mu, logvar, kld_weight)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += reduce_tensor(loss.detach()).item()

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    kld_weight: float,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    for images in val_loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            recon, mu, logvar = model(images)
            loss, _, _ = vae_loss_fn(recon, images, mu, logvar, kld_weight)

        total_loss += reduce_tensor(loss.detach()).item()

    return total_loss / len(val_loader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vae.yaml")
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device, local_rank = setup_distributed()

    train_loader, val_loader, train_sampler, _ = get_dataloaders(
        data_root=args.data_root,
        batch_size=cfg.get("batch_size", 64),
        num_workers=cfg.get("num_workers", 2),
        val_frac=cfg.get("val_frac", 0.05),
        distributed=is_distributed(),
        in_memory=True,
        image_size=cfg.get("image_size", 128),
    )

    model = VAE(latent_dim=cfg.get("latent_dim", 256)).to(device)
    print_model_info(model, "VAE")
    model = wrap_ddp(model, local_rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 0.0005))
    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")
    epochs = cfg.get("epochs", 50)
    save_dir = Path(cfg.get("save_dir", "checkpoints/vae"))

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            cfg.get("kld_weight", 0.00025),
            device,
        )

        val_loss = 0.0
        if val_loader is not None:
            val_loss = evaluate(
                model,
                val_loader,
                cfg.get("kld_weight", 0.00025),
                device,
            )

        if is_main_process():
            print(
                f"[Epoch {epoch:03d}/{epochs:03d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            if epoch % cfg.get("save_interval", 5) == 0 or is_best:
                model.eval()
                with torch.no_grad():
                    raw_model = model.module if hasattr(model, "module") else model
                    z = torch.randn(64, cfg.get("latent_dim", 256), device=device)
                    samples = raw_model.decode(z)
                    save_samples(samples, f"samples/vae/epoch_{epoch:03d}.png")

                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model": model,
                        "optimizer": optimizer,
                        "scaler": scaler,
                        "config": cfg,
                    },
                    is_best=is_best,
                    checkpoint_dir=save_dir,
                )

    cleanup_distributed()


if __name__ == "__main__":
    main()
