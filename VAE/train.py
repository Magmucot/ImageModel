"""
Скрипт обучения β-VAE на датасете FFHQ 128×128.

Запуск:
    python VAE/train.py --data_root ./data --epochs 50 --batch_size 32
    python VAE/train.py --config configs/vae.yaml
    python VAE/train.py --resume checkpoints/vae/checkpoint_ep020.pt

Опции:
    --dry_run   — 2 мини-эпохи без реальных данных (для проверки кода)
"""

import os
import sys
import argparse
import yaml
import math
from pathlib import Path

# Добавляем корень проекта в sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from VAE.vae import VAE, vae_loss
from utils.utils import (
    TrainingLogger, Visualizer,
    save_checkpoint, load_checkpoint,
    save_sample_grid, print_model_info,
)


# ─────────────────────────────────────────────────────────────────────────────
# Аргументы
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="β-VAE Training on FFHQ 128×128")

    # Конфиг
    parser.add_argument("--config", type=str, default="configs/vae.yaml",
                        help="путь к YAML конфигу (перекрывается CLI аргументами)")

    # Данные
    parser.add_argument("--data_root",   type=str,   default="./data")
    parser.add_argument("--val_frac",    type=float, default=0.05)
    parser.add_argument("--num_workers", type=int,   default=4)

    # Модель
    parser.add_argument("--latent_dim",  type=int,   default=256)
    parser.add_argument("--beta",        type=float, default=4.0)

    # Обучение
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--weight_decay",type=float, default=1e-5)
    parser.add_argument("--grad_clip",   type=float, default=1.0)
    parser.add_argument("--warmup_epochs",type=int,  default=2)

    # Логирование / чекпоинты
    parser.add_argument("--output_dir",  type=str,   default="checkpoints/vae")
    parser.add_argument("--save_every",  type=int,   default=5)
    parser.add_argument("--sample_every",type=int,   default=5)
    parser.add_argument("--log_every",   type=int,   default=20)
    parser.add_argument("--n_samples",   type=int,   default=64)

    # Прочее
    parser.add_argument("--device",      type=str,   default="auto")
    parser.add_argument("--resume",      type=str,   default=None,
                        help="путь к чекпоинту для продолжения обучения")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--dry_run",     action="store_true",
                        help="2 мини-эпохи без реальных данных")

    args = parser.parse_args()

    # Загружаем YAML и перезаписываем дефолты (если файл существует)
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
# Синтетический датасет для dry_run
# ─────────────────────────────────────────────────────────────────────────────

class DummyDataLoader:
    """Итерируется по случайным тензорам — для теста без данных."""
    def __init__(self, batch_size: int = 8, n_batches: int = 5, device: str = "cpu"):
        self.batch_size = batch_size
        self.n_batches  = n_batches
        self.device     = device

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            yield torch.randn(self.batch_size, 3, 128, 128)


# ─────────────────────────────────────────────────────────────────────────────
# LR Scheduler с warmup
# ─────────────────────────────────────────────────────────────────────────────

def get_scheduler(optimizer, args, n_epochs: int):
    """Cosine annealing с linear warmup."""
    if args.warmup_epochs > 0:
        from torch.optim.lr_scheduler import LinearLR, SequentialLR
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                          total_iters=args.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, n_epochs - args.warmup_epochs),
                                   eta_min=args.lr * 0.01)
        return SequentialLR(optimizer, [warmup, cosine],
                            milestones=[args.warmup_epochs])
    return CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=args.lr * 0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Эпоха обучения
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:     VAE,
    loader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    args:      argparse.Namespace,
    logger:    TrainingLogger,
    epoch:     int,
) -> dict:
    """Один проход по тренировочному датасету."""
    model.train()
    total_loss_acc  = 0.0
    recon_loss_acc  = 0.0
    kl_loss_acc     = 0.0

    for step, batch in enumerate(loader):
        imgs = batch.to(device, non_blocking=True)

        optimizer.zero_grad()

        recon, mu, logvar = model(imgs)
        total, recon_l, kl_l = vae_loss(recon, imgs, mu, logvar, beta=model.beta)

        total.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        total_loss_acc += total.item()
        recon_loss_acc += recon_l.item()
        kl_loss_acc    += kl_l.item()

        if (step + 1) % args.log_every == 0:
            logger.log(
                epoch=epoch, step=step + 1,
                total_loss=total.item(),
                recon_loss=recon_l.item(),
                kl_loss=kl_l.item(),
            )
            logger.print_step(
                epoch, args.epochs, step + 1, len(loader),
                total=total.item(), recon=recon_l.item(), kl=kl_l.item(),
            )

    n = len(loader)
    return {
        "total_loss": total_loss_acc / n,
        "recon_loss": recon_loss_acc / n,
        "kl_loss":    kl_loss_acc    / n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Валидационная эпоха
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(
    model:  VAE,
    loader,
    device: torch.device,
    args:   argparse.Namespace,
) -> dict:
    """Один проход по валидационному датасету."""
    model.eval()
    total_acc = recon_acc = kl_acc = 0.0

    for batch in loader:
        imgs = batch.to(device, non_blocking=True)
        recon, mu, logvar = model(imgs)
        total, recon_l, kl_l = vae_loss(recon, imgs, mu, logvar, beta=model.beta)
        total_acc += total.item()
        recon_acc += recon_l.item()
        kl_acc    += kl_l.item()

    n = max(len(loader), 1)
    return {
        "val_total":    total_acc / n,
        "val_recon":    recon_acc / n,
        "val_kl":       kl_acc    / n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Воспроизводимость
    torch.manual_seed(args.seed)

    # Устройство
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            device = torch.nn.DataParallel(device)
            
    else:
        device = torch.device(args.device)
    print(f"\n🚀 β-VAE Training | Device: {device} | β={args.beta}")

    # ── Данные ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print("⚠️  DRY RUN — используются синтетические данные")
        train_loader = DummyDataLoader(batch_size=args.batch_size, n_batches=10)
        val_loader   = DummyDataLoader(batch_size=args.batch_size, n_batches=3)
        args.epochs  = 2
    else:
        from data.dataset import get_dataloaders
        train_loader, val_loader = get_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_frac=args.val_frac,
        )

    # ── Модель ────────────────────────────────────────────────────────────────
    model = VAE(latent_dim=args.latent_dim, beta=args.beta).to(device)
    print_model_info(model, f"β-VAE  (latent_dim={args.latent_dim}, β={args.beta})")

    # ── Оптимизатор + Scheduler ───────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, args, args.epochs)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val    = float("inf")
    if args.resume:
        ckpt = load_checkpoint(args.resume, device=str(device))
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val    = ckpt.get("best_val", float("inf"))

    # ── Логгер и визуализатор ─────────────────────────────────────────────────
    logger = TrainingLogger(args.output_dir, model_name="VAE")
    vis    = Visualizer(args.output_dir, model_name="β-VAE")

    print(f"\n{'═' * 60}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"{'═' * 60}\n")

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):

        # Обучение
        train_metrics = train_epoch(model, train_loader, optimizer, device, args, logger, epoch)

        # Валидация
        val_metrics = val_epoch(model, val_loader, device, args)

        # LR step
        scheduler.step()

        # Логируем эпоху
        epoch_metrics = {**train_metrics, **val_metrics}
        logger.log_epoch(epoch=epoch, **epoch_metrics)
        logger.print_epoch_summary(epoch=epoch, **epoch_metrics)

        is_best = val_metrics["val_total"] < best_val
        if is_best:
            best_val = val_metrics["val_total"]

        # ── Чекпоинт ─────────────────────────────────────────────────────────
        if epoch % args.save_every == 0 or epoch == args.epochs:
            state = {
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val":  best_val,
                "args":      vars(args),
            }
            save_checkpoint(
                state, args.output_dir,
                filename=f"checkpoint_ep{epoch:03d}.pt",
                is_best=is_best,
            )

        # ── Образцы ──────────────────────────────────────────────────────────
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            with torch.no_grad():
                # Сгенерированные из латентного пространства
                samples = model.sample(n=args.n_samples, device=str(device))
                save_sample_grid(
                    samples,
                    path=f"{args.output_dir}/samples/gen_ep{epoch:03d}.png",
                    nrow=8,
                    title=f"β-VAE Generated — Epoch {epoch}",
                )
                # Реконструкции (первый батч из val)
                val_batch = next(iter(val_loader))
                if isinstance(val_batch, torch.Tensor):
                    val_batch = val_batch[:8].to(device)
                    recon_val, _, _ = model(val_batch)
                    comparison = torch.cat([val_batch[:8], recon_val[:8]])
                    save_sample_grid(
                        comparison,
                        path=f"{args.output_dir}/samples/recon_ep{epoch:03d}.png",
                        nrow=8,
                        title=f"β-VAE Reconstructions — Epoch {epoch}",
                    )

        # ── Графики loss ──────────────────────────────────────────────────────
        if epoch % args.save_every == 0 or epoch == args.epochs:
            vis.plot_curves(logger.epoch_history, epoch=epoch, save=True)

    # ── Финальный постер ──────────────────────────────────────────────────────
    with torch.no_grad():
        final_samples = model.sample(n=args.n_samples, device=str(device))
    vis.plot_final_summary(
        logger.epoch_history,
        samples=final_samples,
        extra_info=f"β={args.beta}",
    )

    logger.close()
    print(f"\n✅ Обучение завершено! Результаты → {args.output_dir}/")


if __name__ == "__main__":
    main()
