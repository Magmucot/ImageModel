"""Обучение GAN на 2x T4 (DDP, AMP FP16, In-Memory Dataset)."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import get_dataloaders
from GAN.models import Discriminator, Generator
from utils.distributed import (
    cleanup_distributed,
    get_rank,
    is_distributed,
    is_main_process,
    reduce_tensor,
    setup_distributed,
    wrap_ddp,
)
from utils.utils import load_config, print_model_info, save_checkpoint, save_samples


def train_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scaler_g: torch.amp.GradScaler,
    scaler_d: torch.amp.GradScaler,
    criterion: nn.Module,
    latent_dim: int,
    device: torch.device,
) -> tuple[float, float]:
    generator.train()
    discriminator.train()

    total_loss_g = 0.0
    total_loss_d = 0.0

    for real_images in train_loader:
        batch_size = real_images.size(0)
        real_images = real_images.to(device, non_blocking=True)

        real_labels = torch.ones((batch_size, 1), device=device)
        fake_labels = torch.zeros((batch_size, 1), device=device)

        # ---------------------
        # 1. Шаг Дискриминатора
        # ---------------------
        optimizer_d.zero_grad(set_to_none=True)
        z = torch.randn(batch_size, latent_dim, device=device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            fake_images = generator(z)
            d_real_out = discriminator(real_images)
            d_fake_out = discriminator(fake_images.detach())

            loss_d_real = criterion(d_real_out, real_labels)
            loss_d_fake = criterion(d_fake_out, fake_labels)
            loss_d = (loss_d_real + loss_d_fake) * 0.5

        scaler_d.scale(loss_d).backward()
        scaler_d.step(optimizer_d)
        scaler_d.update()

        # ---------------------
        # 2. Шаг Генератора
        # ---------------------
        optimizer_g.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            # Переключение requires_grad НЕ делается, чтобы не ломать reducer DDP
            d_g_out = discriminator(fake_images)
            loss_g = criterion(d_g_out, real_labels)

        scaler_g.scale(loss_g).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()

        total_loss_g += reduce_tensor(loss_g.detach()).item()
        total_loss_d += reduce_tensor(loss_d.detach()).item()

    num_batches = len(train_loader)
    return total_loss_g / num_batches, total_loss_d / num_batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gan.yaml")
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device, local_rank = setup_distributed()

    train_loader, _, train_sampler, _ = get_dataloaders(
        data_root=args.data_root,
        batch_size=cfg.get("batch_size", 64),
        num_workers=cfg.get("num_workers", 2),
        val_frac=0.0,
        distributed=is_distributed(),
        in_memory=True,
        image_size=cfg.get("image_size", 128),
    )

    latent_dim = cfg.get("latent_dim", 256)
    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)

    print_model_info(generator, "Generator")
    print_model_info(discriminator, "Discriminator")

    generator = wrap_ddp(generator, local_rank)
    discriminator = wrap_ddp(discriminator, local_rank)

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=cfg.get("lr_g", 0.0002),
        betas=(cfg.get("beta1", 0.5), 0.999),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=cfg.get("lr_d", 0.0002),
        betas=(cfg.get("beta1", 0.5), 0.999),
    )

    scaler_g = torch.amp.GradScaler("cuda")
    scaler_d = torch.amp.GradScaler("cuda")
    criterion = nn.BCEWithLogitsLoss()

    fixed_noise = torch.randn(64, latent_dim, device=device)
    epochs = cfg.get("epochs", 100)
    save_dir = Path(cfg.get("save_dir", "checkpoints/gan"))

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        loss_g, loss_d = train_epoch(
            generator,
            discriminator,
            train_loader,
            optimizer_g,
            optimizer_d,
            scaler_g,
            scaler_d,
            criterion,
            latent_dim,
            device,
        )

        if is_main_process():
            print(
                f"[Epoch {epoch:03d}/{epochs:03d}] Loss_G: {loss_g:.4f} | Loss_D: {loss_d:.4f}"
            )

            if epoch % cfg.get("save_interval", 5) == 0 or epoch == epochs:
                generator.eval()
                with torch.no_grad():
                    raw_g = generator.module if hasattr(generator, "module") else generator
                    samples = raw_g(fixed_noise)
                    save_samples(samples, f"samples/gan/epoch_{epoch:03d}.png")

                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model": generator,
                        "optimizer": optimizer_g,
                        "config": cfg,
                    },
                    is_best=False,
                    checkpoint_dir=save_dir,
                    filename=f"checkpoint_epoch_{epoch:03d}.pt",
                )

    cleanup_distributed()


if __name__ == "__main__":
    main()
