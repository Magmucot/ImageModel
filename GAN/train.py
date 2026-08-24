"""Обучение GAN на 2x T4 (DDP, AMP FP16, In-Memory Dataset)."""

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
from GAN.gan import Discriminator, Generator, smooth_labels, weights_init
from utils.distributed import (
    cleanup_distributed,
    is_main_process,
    reduce_tensor,
    setup_distributed,
    unwrap_model,
    wrap_ddp,
)
from utils.utils import (
    get_config_value,
    load_config,
    print_model_info,
    save_checkpoint,
    save_samples,
)


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
    label_smooth: float = 0.0,
    n_critic: int = 1,
) -> tuple[float, float]:
    """
    Один epoch обучения GAN.

    DDP-инвариант: каждый DDP-модуль за итерацию получает ровно один
    forward + backward по своим параметрам:
      - шаг D: fake генерируются через unwrapped Generator (без хуков DDP-G);
      - шаг G: дискриминатор вызывается через unwrapped Discriminator
        (без хуков DDP-D), градиенты синхронизируются только для G.
    """
    generator.train()
    discriminator.train()

    raw_generator = unwrap_model(generator)
    raw_discriminator = unwrap_model(discriminator)

    total_loss_g = 0.0
    total_loss_d = 0.0
    n_steps = 0
    loss_d_value = 0.0

    critic_step = 0

    for real_images in train_loader:
        batch_size = real_images.size(0)
        real_images = real_images.to(device, non_blocking=True)

        # ---------------------
        # 1. Шаг(ы) Дискриминатора (n_critic на один шаг G)
        # ---------------------
        for _ in range(n_critic):
            optimizer_d.zero_grad(set_to_none=True)
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                # Unwrapped G: без редуктора DDP-G (градиенты G здесь не нужны)
                fake_images = raw_generator(z)

                d_real_out = discriminator(real_images)
                d_fake_out = discriminator(fake_images.detach())

                if label_smooth > 0.0:
                    real_targets = smooth_labels(
                        batch_size, real=True, smooth=label_smooth, device=str(device)
                    )
                    fake_targets = smooth_labels(
                        batch_size, real=False, smooth=label_smooth, device=str(device)
                    )
                else:
                    real_targets = torch.ones(batch_size, device=device)
                    fake_targets = torch.zeros(batch_size, device=device)

                loss_d_real = criterion(d_real_out.float(), real_targets)
                loss_d_fake = criterion(d_fake_out.float(), fake_targets)
                loss_d = (loss_d_real + loss_d_fake) * 0.5

            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            loss_d_value = loss_d.detach().float()

        critic_step += n_critic

        # ---------------------
        # 2. Шаг Генератора (раз в n_critic шагов D)
        # ---------------------
        if critic_step >= n_critic:
            critic_step = 0
            optimizer_g.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                fake_images = generator(z)

                g_targets = torch.ones(batch_size, device=device)

                # Unwrapped D: без редуктора DDP-D (градиенты D здесь не нужны)
                d_g_out = raw_discriminator(fake_images)
                loss_g = criterion(d_g_out.float(), g_targets)

            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            total_loss_g += reduce_tensor(loss_g.detach().float()).item()
            n_steps += 1

        total_loss_d += reduce_tensor(loss_d_value).item()

    num_batches = max(len(train_loader), 1)
    return total_loss_g / max(n_steps, 1), total_loss_d / num_batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gan.yaml")
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device, local_rank, _, distributed = setup_distributed()

    train_loader, _, train_sampler, _ = get_dataloaders(
        data_root=args.data_root,
        batch_size=get_config_value(cfg, "batch_size", 32),
        num_workers=get_config_value(cfg, "num_workers", 4),
        val_frac=0.0,
        distributed=distributed,
        in_memory=get_config_value(cfg, "in_memory", False),
        image_size=get_config_value(cfg, "image_size", 128),
    )

    latent_dim = get_config_value(cfg, "latent_dim", 100)
    label_smooth = get_config_value(cfg, "label_smooth", 0.0)
    n_critic = max(1, int(get_config_value(cfg, "n_critic", 1)))

    generator = Generator(
        latent_dim=latent_dim,
        ngf=get_config_value(cfg, "ngf", 64),
    ).to(device)
    discriminator = Discriminator(
        ndf=get_config_value(cfg, "ndf", 64),
    ).to(device)

    print_model_info(generator, "Generator")
    print_model_info(discriminator, "Discriminator")

    generator = wrap_ddp(generator, device, local_rank, distributed)
    discriminator = wrap_ddp(discriminator, device, local_rank, distributed)

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=get_config_value(cfg, "lr_g", 0.0002),
        betas=(
            get_config_value(cfg, "beta1", 0.5),
            get_config_value(cfg, "beta2", 0.999),
        ),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=get_config_value(cfg, "lr_d", 0.0002),
        betas=(
            get_config_value(cfg, "beta1", 0.5),
            get_config_value(cfg, "beta2", 0.999),
        ),
    )

    use_cuda = device.type == "cuda"
    scaler_g = torch.amp.GradScaler("cuda", enabled=use_cuda)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_cuda)
    # BCEWithLogitsLoss безопасен под autocast (в отличие от BCELoss).
    criterion = nn.BCEWithLogitsLoss()

    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
    epochs = get_config_value(cfg, "epochs", 100)
    save_dir = Path(get_config_value(cfg, "output_dir", "checkpoints/gan"))

    best_loss_g = float("inf")

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
            label_smooth=label_smooth,
            n_critic=n_critic,
        )

        if is_main_process():
            print(
                f"[Epoch {epoch:03d}/{epochs:03d}] Loss_G: {loss_g:.4f} | Loss_D: {loss_d:.4f}"
            )

            is_best = loss_g < best_loss_g
            if is_best:
                best_loss_g = loss_g

            if (
                epoch % get_config_value(cfg, "save_every", 10) == 0
                or is_best
                or epoch == epochs
            ):
                generator.eval()
                with torch.no_grad():
                    raw_g = unwrap_model(generator)
                    samples = raw_g(fixed_noise)
                    save_samples(samples, save_dir / f"samples/epoch_{epoch:03d}.png")

                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model": generator,
                        "optimizer": optimizer_g,
                        "config": cfg,
                        "args": {
                            "latent_dim": latent_dim,
                            "ngf": get_config_value(cfg, "ngf", 64),
                        },
                    },
                    is_best=is_best,
                    checkpoint_dir=save_dir,
                    filename=f"checkpoint_epoch_{epoch:03d}.pt",
                )

    cleanup_distributed()


if __name__ == "__main__":
    main()
