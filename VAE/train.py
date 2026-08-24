"""Обучение VAE на 2x T4 (DDP, AMP FP16, In-Memory Dataset)."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import get_dataloaders
from utils.distributed import (
    cleanup_distributed,
    is_main_process,
    reduce_tensor,
    setup_distributed,
    wrap_ddp,
)
from utils.utils import (
    get_config_value,
    load_config,
    print_model_info,
    save_checkpoint,
    save_samples,
)
from VAE.vae import VAE


def vae_loss_fn(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kld_loss
    return total_loss, recon_loss, kld_loss


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    beta: float,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images in train_loader:
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            recon, mu, logvar = model(images)
            loss, _, _ = vae_loss_fn(recon, images, mu, logvar, beta)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += reduce_tensor(loss.detach()).item()

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    beta: float,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    for images in val_loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            recon, mu, logvar = model(images)
            loss, _, _ = vae_loss_fn(recon, images, mu, logvar, beta)

        total_loss += reduce_tensor(loss.detach()).item()

    return total_loss / len(val_loader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vae.yaml")
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device, local_rank, _, distributed = setup_distributed()

    train_loader, val_loader, train_sampler, _ = get_dataloaders(
        data_root=args.data_root,
        batch_size=get_config_value(cfg, "batch_size", 32),
        num_workers=get_config_value(cfg, "num_workers", 4),
        val_frac=get_config_value(cfg, "val_frac", 0.05),
        distributed=distributed,
        in_memory=get_config_value(cfg, "in_memory", False),
        image_size=get_config_value(cfg, "image_size", 128),
    )

    latent_dim = get_config_value(cfg, "latent_dim", 256)
    model = VAE(
        latent_dim=latent_dim,
        beta=get_config_value(cfg, "beta", 4.0),
    ).to(device)
    print_model_info(model, "VAE")
    model = wrap_ddp(model, device, local_rank, distributed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=get_config_value(cfg, "lr", 1e-4),
        weight_decay=get_config_value(cfg, "weight_decay", 1e-5),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    epochs = get_config_value(cfg, "epochs", 50)
    save_dir = Path(get_config_value(cfg, "output_dir", "checkpoints/vae"))

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            get_config_value(cfg, "beta", 4.0),
            device,
        )

        val_loss = 0.0
        if val_loader is not None:
            val_loss = evaluate(
                model,
                val_loader,
                get_config_value(cfg, "beta", 4.0),
                device,
            )

        if is_main_process():
            print(
                f"[Epoch {epoch:03d}/{epochs:03d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            if epoch % get_config_value(cfg, "save_every", 5) == 0 or is_best:
                model.eval()
                with torch.no_grad():
                    raw_model = model.module if hasattr(model, "module") else model
                    n_samples = get_config_value(cfg, "n_samples", 64)
                    z = torch.randn(n_samples, latent_dim, device=device)
                    samples = raw_model.decode(z)
                    save_samples(samples, save_dir / f"samples/epoch_{epoch:03d}.png")

                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model": model,
                        "optimizer": optimizer,
                        "scaler": scaler,
                        "config": cfg,
                        "args": {
                            "latent_dim": latent_dim,
                            "beta": get_config_value(cfg, "beta", 4.0),
                        },
                    },
                    is_best=is_best,
                    checkpoint_dir=save_dir,
                )

    cleanup_distributed()


if __name__ == "__main__":
    main()
