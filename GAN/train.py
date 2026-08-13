"""
Скрипт обучения DCGAN (Spectral Norm + Self-Attention) на FFHQ 128×128.

Запуск:
    python GAN/train.py --data_root ./data --epochs 100 --batch_size 32
    python GAN/train.py --config configs/gan.yaml
    python GAN/train.py --resume checkpoints/gan/checkpoint_ep050.pt

Опции:
    --dry_run   — 2 мини-эпохи без реальных данных (для проверки кода)

Особенности:
  - Label smoothing: реальные метки [0.9, 1.0], фейковые [0.0, 0.1]
  - Сохраняет D_loss, G_loss, D(x), D(G(z)) в CSV
  - Предупреждает о возможном mode collapse (D(x) ≈ 0 или G слишком слаб)
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

from GAN.gan import Generator, Discriminator, weights_init, smooth_labels
from utils.utils import (
    TrainingLogger, Visualizer,
    save_checkpoint, load_checkpoint,
    save_sample_grid, print_model_info,
)


# ─────────────────────────────────────────────────────────────────────────────
# Аргументы
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DCGAN Training on FFHQ 128×128")

    parser.add_argument("--config",      type=str,   default="configs/gan.yaml")
    parser.add_argument("--data_root",   type=str,   default="./data")
    parser.add_argument("--val_frac",    type=float, default=0.05)
    parser.add_argument("--num_workers", type=int,   default=4)

    parser.add_argument("--latent_dim",  type=int,   default=100)
    parser.add_argument("--ngf",         type=int,   default=64)
    parser.add_argument("--ndf",         type=int,   default=64)

    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr_g",        type=float, default=2e-4)
    parser.add_argument("--lr_d",        type=float, default=2e-4)
    parser.add_argument("--beta1",       type=float, default=0.5)
    parser.add_argument("--beta2",       type=float, default=0.999)
    parser.add_argument("--label_smooth",type=float, default=0.1)
    parser.add_argument("--n_critic",    type=int,   default=1,
                        help="шагов D на 1 шаг G")

    parser.add_argument("--output_dir",  type=str,   default="checkpoints/gan")
    parser.add_argument("--save_every",  type=int,   default=10)
    parser.add_argument("--sample_every",type=int,   default=5)
    parser.add_argument("--log_every",   type=int,   default=20)
    parser.add_argument("--n_samples",   type=int,   default=64)

    parser.add_argument("--device",      type=str,   default="auto")
    parser.add_argument("--resume",      type=str,   default=None)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--dry_run",     action="store_true")

    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for section in cfg.values():
            if isinstance(section, dict):
                for k, v in section.items():
                    if hasattr(args, k) and getattr(args, k) == parser.get_default(k):
                        setattr(args, k, v)

    return args


# ─────────────────────────────────────────────────────────────────────────────
# Синтетический DataLoader для dry_run
# ─────────────────────────────────────────────────────────────────────────────

class DummyDataLoader:
    def __init__(self, batch_size=8, n_batches=10):
        self.batch_size = batch_size
        self.n_batches  = n_batches

    def __len__(self): return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            yield torch.randn(self.batch_size, 3, 128, 128)


# ─────────────────────────────────────────────────────────────────────────────
# Эпоха обучения GAN
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    G, D, loader,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    logger: TrainingLogger,
    epoch: int,
    fixed_noise: torch.Tensor,
) -> dict:
    G.train(); D.train()

    d_loss_acc = g_loss_acc = 0.0
    dx_acc     = dg_acc1    = dg_acc2 = 0.0
    n_steps = 0

    for step, real in enumerate(loader):
        real = real.to(device, non_blocking=True)
        B    = real.size(0)
        dev  = str(device)

        # ── Обучение Discriminator ────────────────────────────────────────
        for _ in range(args.n_critic):
            D.zero_grad()

            # Реальные изображения
            real_labels = smooth_labels(B, real=True,  smooth=args.label_smooth, device=dev)
            d_real = D(real)
            loss_d_real = criterion(d_real, real_labels)

            # Фейковые изображения
            z = torch.randn(B, args.latent_dim, 1, 1, device=device)
            fake = G(z).detach()
            fake_labels = smooth_labels(B, real=False, smooth=args.label_smooth, device=dev)
            d_fake = D(fake)
            loss_d_fake = criterion(d_fake, fake_labels)

            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            opt_d.step()

        # ── Обучение Generator ────────────────────────────────────────────
        G.zero_grad()
        z    = torch.randn(B, args.latent_dim, 1, 1, device=device)
        fake = G(z)
        # G хочет чтобы D считал фейки реальными
        g_labels = torch.ones(B, device=device)
        d_g_z2   = D(fake)
        loss_g   = criterion(d_g_z2, g_labels)
        loss_g.backward()
        opt_g.step()

        # Аккумуляция статистик
        d_loss_acc += loss_d.item()
        g_loss_acc += loss_g.item()
        dx_acc     += d_real.mean().item()
        dg_acc1    += d_fake.mean().item()
        dg_acc2    += d_g_z2.mean().item()
        n_steps    += 1

        if (step + 1) % args.log_every == 0:
            logger.log(
                epoch=epoch, step=step + 1,
                d_loss=loss_d.item(), g_loss=loss_g.item(),
                D_x=d_real.mean().item(), D_G_z1=d_fake.mean().item(),
                D_G_z2=d_g_z2.mean().item(),
            )
            logger.print_step(
                epoch, args.epochs, step + 1, len(loader),
                D_loss=loss_d.item(), G_loss=loss_g.item(),
                D_x=d_real.mean().item(), D_Gz=d_g_z2.mean().item(),
            )

    n = max(n_steps, 1)
    return {
        "d_loss":  d_loss_acc / n,
        "g_loss":  g_loss_acc / n,
        "D_x":     dx_acc     / n,
        "D_G_z1":  dg_acc1    / n,
        "D_G_z2":  dg_acc2    / n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n🚀 DCGAN Training | Device: {device}")

    # ── Данные ────────────────────────────────────────────────────────────────
    if args.dry_run:
        print("⚠️  DRY RUN — используются синтетические данные")
        train_loader = DummyDataLoader(batch_size=args.batch_size, n_batches=10)
        args.epochs  = 2
    else:
        from data.dataset import get_dataloaders
        train_loader, _ = get_dataloaders(
            data_root=args.data_root, batch_size=args.batch_size,
            num_workers=args.num_workers, val_frac=args.val_frac,
        )

    # ── Модели ────────────────────────────────────────────────────────────────
    G = Generator(latent_dim=args.latent_dim, ngf=args.ngf).to(device)
    D = Discriminator(nc=3, ndf=args.ndf).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    print_model_info(G, f"Generator  (latent_dim={args.latent_dim})")
    print_model_info(D, "Discriminator (Spectral Norm)")

    # ── Оптимизаторы ──────────────────────────────────────────────────────────
    opt_g = optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d = optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    criterion = nn.BCELoss()

    # Фиксированный шум для мониторинга прогресса
    fixed_noise = torch.randn(args.n_samples, args.latent_dim, 1, 1, device=device)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume:
        ckpt = load_checkpoint(args.resume, device=str(device))
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = ckpt.get("epoch", 0) + 1

    # ── Логгер и визуализатор ─────────────────────────────────────────────────
    logger = TrainingLogger(args.output_dir, model_name="GAN")
    vis    = Visualizer(args.output_dir, model_name="DCGAN")

    print(f"\n{'═' * 60}")
    print(f"  Epochs: {args.epochs}  |  Batch: {args.batch_size}")
    print(f"  LR_G: {args.lr_g}  |  LR_D: {args.lr_d}")
    print(f"  Label smooth: {args.label_smooth}")
    print(f"{'═' * 60}\n")

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):

        metrics = train_epoch(
            G, D, train_loader, opt_g, opt_d, criterion,
            device, args, logger, epoch, fixed_noise,
        )
        logger.log_epoch(epoch=epoch, **metrics)
        logger.print_epoch_summary(epoch=epoch, **metrics)

        # Предупреждение о mode collapse
        if metrics["D_x"] < 0.3:
            print("  ⚠️  D(x) < 0.3 — D слишком слабый, рассмотри снижение lr_d")
        if metrics["D_G_z2"] > 0.8:
            print("  ⚠️  D(G(z)) > 0.8 — возможен mode collapse")

        # ── Образцы ──────────────────────────────────────────────────────────
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            G.eval()
            with torch.no_grad():
                fake_imgs = G(fixed_noise)
            save_sample_grid(
                fake_imgs,
                path=f"{args.output_dir}/samples/gen_ep{epoch:03d}.png",
                nrow=8,
                title=f"DCGAN Generated — Epoch {epoch}",
            )
            G.train()

        # ── Чекпоинт ─────────────────────────────────────────────────────────
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(
                {
                    "epoch": epoch, "G": G.state_dict(), "D": D.state_dict(),
                    "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                    "args":  vars(args),
                },
                args.output_dir,
                filename=f"checkpoint_ep{epoch:03d}.pt",
            )

        # ── Графики ───────────────────────────────────────────────────────────
        if epoch % args.save_every == 0 or epoch == args.epochs:
            vis.plot_curves(logger.epoch_history, epoch=epoch, save=True)

    # ── Финальный постер ──────────────────────────────────────────────────────
    G.eval()
    with torch.no_grad():
        final_samples = G(fixed_noise)
    vis.plot_final_summary(logger.epoch_history, samples=final_samples)

    logger.close()
    print(f"\n✅ Обучение завершено! Результаты → {args.output_dir}/")


if __name__ == "__main__":
    main()
