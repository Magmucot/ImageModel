"""
Скрипт обучения DDPM (UNet + Self-Attention + Cosine Schedule) на FFHQ 128×128.

Запуск:
    python DDPM/train.py --data_root ./data --epochs 200 --batch_size 16
    python DDPM/train.py --config configs/ddpm.yaml
    python DDPM/train.py --resume checkpoints/ddpm/checkpoint_ep100.pt

Опции:
    --dry_run   — 2 мини-эпохи без реальных данных
    --ema_decay — EMA весов модели (рекомендуется 0.9999 для DDPM)

Замечание:
    Один шаг семплирования DDPM требует T forward-пассов через UNet,
    поэтому sample_every лучше ставить 20+.
    Для быстрого семплинга в ноутбуке используйте DDIM (не реализовано здесь,
    но легко добавить поверх той же модели).
"""

import os
import sys
import copy
import argparse
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from DDPM.ddpm import UNet, DDPM
from utils.utils import (
    TrainingLogger, Visualizer,
    save_checkpoint, load_checkpoint,
    save_sample_grid, print_model_info,
)


# ─────────────────────────────────────────────────────────────────────────────
# Exponential Moving Average
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    """
    Экспоненциальное скользящее среднее весов модели.
    Значительно улучшает качество генерации DDPM без затрат на обучение.

    Использование:
        ema = EMA(model, decay=0.9999)
        # после каждого шага оптимизатора:
        ema.update(model)
        # для семплирования:
        with ema.average_parameters():
            samples = ddpm.sample(...)
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay    = decay
        self.shadow   = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Обновляет EMA-веса после шага оптимизатора."""
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow.load_state_dict(state_dict)


# ─────────────────────────────────────────────────────────────────────────────
# Аргументы
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDPM Training on FFHQ 128×128")

    parser.add_argument("--config",        type=str,   default="configs/ddpm.yaml")
    parser.add_argument("--data_root",     type=str,   default="./data")
    parser.add_argument("--val_frac",      type=float, default=0.05)
    parser.add_argument("--num_workers",   type=int,   default=4)

    parser.add_argument("--base_channels", type=int,   default=64)
    parser.add_argument("--time_emb_dim",  type=int,   default=256)
    parser.add_argument("--timesteps",     type=int,   default=1000)
    parser.add_argument("--schedule",      type=str,   default="cosine",
                        choices=["cosine", "linear"])
    parser.add_argument("--dropout",       type=float, default=0.1)

    parser.add_argument("--epochs",        type=int,   default=200)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--weight_decay",  type=float, default=0.0)
    parser.add_argument("--grad_clip",     type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int,   default=5)
    parser.add_argument("--ema_decay",     type=float, default=0.9999,
                        help="EMA decay для весов модели (0 = отключено)")

    parser.add_argument("--output_dir",   type=str,   default="checkpoints/ddpm")
    parser.add_argument("--save_every",   type=int,   default=10)
    parser.add_argument("--sample_every", type=int,   default=20,
                        help="семплирование DDPM медленное — ставьте 20+")
    parser.add_argument("--log_every",    type=int,   default=10)
    parser.add_argument("--n_samples",    type=int,   default=16)

    parser.add_argument("--device",       type=str,   default="auto")
    parser.add_argument("--resume",       type=str,   default=None)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--dry_run",      action="store_true")

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
# Синтетический DataLoader
# ─────────────────────────────────────────────────────────────────────────────

class DummyDataLoader:
    def __init__(self, batch_size=4, n_batches=10):
        self.batch_size = batch_size
        self.n_batches  = n_batches

    def __len__(self): return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            yield torch.randn(self.batch_size, 3, 128, 128)


# ─────────────────────────────────────────────────────────────────────────────
# LR Scheduler с warmup
# ─────────────────────────────────────────────────────────────────────────────

def get_scheduler(optimizer, args):
    if args.warmup_epochs > 0:
        from torch.optim.lr_scheduler import LinearLR, SequentialLR
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                          total_iters=args.warmup_epochs)
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs - args.warmup_epochs),
            eta_min=args.lr * 0.01,
        )
        return SequentialLR(optimizer, [warmup, cosine],
                            milestones=[args.warmup_epochs])
    return CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Эпоха обучения
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    ddpm:      DDPM,
    loader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    args:      argparse.Namespace,
    logger:    TrainingLogger,
    epoch:     int,
    ema:       "EMA | None",
) -> dict:
    ddpm.model.train()
    loss_acc = 0.0

    for step, batch in enumerate(loader):
        imgs = batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        loss = ddpm.loss_fn(imgs)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ddpm.model.parameters(), args.grad_clip)

        optimizer.step()

        # Обновляем EMA после каждого шага
        if ema is not None:
            ema.update(
            ddpm.model.module
            if isinstance(ddpm.model, torch.nn.DataParallel)
            else ddpm.model
        )

        loss_acc += loss.item()

        if (step + 1) % args.log_every == 0:
            logger.log(epoch=epoch, step=step + 1, mse_loss=loss.item())
            logger.print_step(
                epoch, args.epochs, step + 1, len(loader),
                mse_loss=loss.item(),
            )

    return {"mse_loss": loss_acc / max(len(loader), 1)}


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

    use_multi_gpu = device.type == "cuda" and torch.cuda.device_count() > 1

    if use_multi_gpu:
        print(f"Using {torch.cuda.device_count()} GPUs")


    # ── Данные ────────────────────────────────────────────────────────────────
    if args.dry_run:
        print("⚠️  DRY RUN — используются синтетические данные")
        train_loader = DummyDataLoader(batch_size=args.batch_size, n_batches=10)
        args.epochs      = 2
        args.sample_every = 2
    else:
        from data.dataset import get_dataloaders
        train_loader, _ = get_dataloaders(
            data_root=args.data_root, batch_size=args.batch_size,
            num_workers=args.num_workers, val_frac=args.val_frac,
        )

    # ── Модель ────────────────────────────────────────────────────────────────
    base_unet = UNet(
        img_channels=3,
        base_channels=args.base_channels,
        time_emb_dim=args.time_emb_dim,
        dropout=args.dropout,
    ).to(device)

    unet = (
        torch.nn.DataParallel(base_unet)
        if use_multi_gpu
        else base_unet
    )

    ddpm = DDPM(
        model=unet,
        timesteps=args.timesteps,
        schedule=args.schedule,
        device=str(device),
    )

ema = EMA(base_unet, decay=args.ema_decay) if args.ema_decay > 0 else None
    print_model_info(
        unet,
        f"UNet DDPM  (base_ch={args.base_channels}, T={args.timesteps}, {args.schedule})",
    )

    # EMA
    ema = EMA(unet, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema:
        print(f"  EMA enabled (decay={args.ema_decay})")

    # ── Оптимизатор + Scheduler ───────────────────────────────────────────────
    optimizer = optim.AdamW(unet.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, args)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_loss   = float("inf")
    if args.resume:
        ckpt = load_checkpoint(args.resume, device=str(device))
        base_unet = (
            unet.module
            if isinstance(unet, torch.nn.DataParallel)
            else unet
        )

        base_unet.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ema and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss   = ckpt.get("best_loss", float("inf"))

    # ── Логгер и визуализатор ─────────────────────────────────────────────────
    logger = TrainingLogger(args.output_dir, model_name="DDPM")
    vis    = Visualizer(args.output_dir,    model_name="DDPM")

    print(f"\n{'═' * 60}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Timesteps:    {args.timesteps}")
    print(f"  Sample every: {args.sample_every} epochs")
    print(f"  Output dir:   {args.output_dir}")
    print(f"{'═' * 60}\n")

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):

        metrics = train_epoch(
            ddpm, train_loader, optimizer, device, args, logger, epoch, ema,
        )
        scheduler.step()

        logger.log_epoch(epoch=epoch, **metrics)
        logger.print_epoch_summary(epoch=epoch, **metrics)

        is_best = metrics["mse_loss"] < best_loss
        if is_best:
            best_loss = metrics["mse_loss"]

        # ── Образцы (используем EMA-модель если доступна) ────────────────────
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            print(f"\n  🎨 Генерируем {args.n_samples} образцов... (T={args.timesteps} шагов)")

            sample_model = ema.shadow if ema else unet
            sample_ddpm  = DDPM(
                sample_model,
                timesteps=args.timesteps,
                schedule=args.schedule,
                device=str(device),
            )
            samples = sample_ddpm.sample(
                n=args.n_samples, img_shape=(3, 128, 128), verbose=True,
            )
            save_sample_grid(
                samples,
                path=f"{args.output_dir}/samples/gen_ep{epoch:03d}.png",
                nrow=4,
                title=f"DDPM Generated — Epoch {epoch}",
            )
            print(f"  ✓ Образцы сохранены → {args.output_dir}/samples/gen_ep{epoch:03d}.png")

        # ── Чекпоинт ─────────────────────────────────────────────────────────
        if epoch % args.save_every == 0 or epoch == args.epochs:
            state = {
                "epoch":     epoch,
                "model": (
                            unet.module.state_dict()
                            if isinstance(unet, torch.nn.DataParallel)
                            else unet.state_dict()
                        ),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
                "args":      vars(args),
            }
            if ema:
                state["ema"] = ema.state_dict()
            save_checkpoint(
                state, args.output_dir,
                filename=f"checkpoint_ep{epoch:03d}.pt",
                is_best=is_best,
            )

        # ── Графики ───────────────────────────────────────────────────────────
        if epoch % args.save_every == 0 or epoch == args.epochs:
            vis.plot_curves(logger.epoch_history, epoch=epoch, save=True)

    # ── Финальный постер ──────────────────────────────────────────────────────
    print("\n  🎨 Финальная генерация для постера...")
    sample_model = ema.shadow if ema else unet
    sample_ddpm  = DDPM(sample_model, timesteps=args.timesteps,
                        schedule=args.schedule, device=str(device))
    final_samples = sample_ddpm.sample(n=args.n_samples, verbose=True)
    vis.plot_final_summary(
        logger.epoch_history,
        samples=final_samples,
        extra_info=f"T={args.timesteps}, {args.schedule}",
    )

    logger.close()
    print(f"\n✅ Обучение завершено! Результаты → {args.output_dir}/")


if __name__ == "__main__":
    main()
