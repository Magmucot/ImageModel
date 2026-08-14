"""
Обучение DDPM.

Single GPU:

    python DDPM/train.py \
        --config configs/ddpm.yaml

Multi GPU:

    torchrun --standalone \
        --nproc_per_node=4 \
        DDPM/train.py \
        --config configs/ddpm.yaml
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from DDPM.ddpm import DDPM, UNet
from utils.distributed import (
    cleanup_distributed,
    is_main_process,
    reduce_mean,
    seed_everything,
    setup_distributed,
    unwrap_model,
    wrap_ddp,
)
from utils.utils import (
    TrainingLogger,
    Visualizer,
    load_checkpoint,
    print_model_info,
    save_checkpoint,
    save_sample_grid,
)


class EMA:
    """EMA настоящей UNet-модели."""

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.9999,
    ):
        self.decay = decay

        self.shadow = copy.deepcopy(
            model
        ).eval()

        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(
        self,
        model: torch.nn.Module,
    ):
        for shadow, current in zip(
            self.shadow.parameters(),
            model.parameters(),
        ):
            shadow.mul_(
                self.decay
            ).add_(
                current,
                alpha=1.0 - self.decay,
            )

        for shadow, current in zip(
            self.shadow.buffers(),
            model.buffers(),
        ):
            shadow.copy_(current)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(
        self,
        state_dict,
    ):
        self.shadow.load_state_dict(
            state_dict
        )


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file) or {}


def get_config_value(
    config: dict,
    key: str,
    default,
):
    aliases = {
        "data_root": ("data", "root"),
        "output_dir": ("logging", "output_dir"),
    }

    if key in aliases:
        section_name, config_key = aliases[key]

        section = config.get(
            section_name,
            {},
        )

        if (
            isinstance(section, dict)
            and config_key in section
        ):
            return section[config_key]

    for section in config.values():
        if not isinstance(section, dict):
            continue

        if key in section:
            return section[key]

    return default


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/ddpm.yaml",
    )

    parser.add_argument(
        "--data_root",
        default=None,
    )

    parser.add_argument(
        "--val_frac",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--pin_memory",
        type=lambda x: x.lower() == "true",
        default=None,
    )

    parser.add_argument(
        "--base_channels",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--time_emb_dim",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--schedule",
        choices=[
            "cosine",
            "linear",
        ],
        default=None,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--grad_clip",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--ema_decay",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        default=None,
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--sample_every",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--log_every",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--resume",
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    defaults = {
        "data_root": "./data",
        "val_frac": 0.05,
        "num_workers": 4,
        "pin_memory": True,
        "base_channels": 64,
        "time_emb_dim": 256,
        "timesteps": 1000,
        "schedule": "cosine",
        "dropout": 0.1,
        "epochs": 200,
        "batch_size": 16,
        "lr": 2e-4,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "warmup_epochs": 5,
        "ema_decay": 0.9999,
        "output_dir": "checkpoints/ddpm",
        "save_every": 10,
        "sample_every": 20,
        "log_every": 10,
        "n_samples": 16,
        "seed": 42,
    }

    for key, default in defaults.items():
        if getattr(args, key) is None:
            setattr(
                args,
                key,
                get_config_value(
                    config,
                    key,
                    default,
                ),
            )

    return args


class DummyDataLoader:
    def __init__(
        self,
        batch_size,
        n_batches=10,
    ):
        self.batch_size = batch_size
        self.n_batches = n_batches

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            yield torch.randn(
                self.batch_size,
                3,
                128,
                128,
            )


def get_scheduler(
    optimizer,
    args,
):
    if args.warmup_epochs > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=args.warmup_epochs,
        )

        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(
                1,
                args.epochs
                - args.warmup_epochs,
            ),
            eta_min=args.lr * 0.01,
        )

        return SequentialLR(
            optimizer,
            [warmup, cosine],
            milestones=[
                args.warmup_epochs
            ],
        )

    return CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )


def train_epoch(
    ddpm,
    loader,
    optimizer,
    device,
    args,
    logger,
    epoch,
    ema,
):
    ddpm.model.train()

    loss_sum = 0.0
    steps = 0

    for step, batch in enumerate(loader):
        images = batch.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss = ddpm.loss_fn(
            images
        )

        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                ddpm.model.parameters(),
                args.grad_clip,
            )

        optimizer.step()

        if ema is not None:
            ema.update(
                unwrap_model(
                    ddpm.model
                )
            )

        loss_sum += loss.detach()
        steps += 1

        if (
            is_main_process()
            and (step + 1) % args.log_every == 0
        ):
            logger.log(
                epoch=epoch,
                step=step + 1,
                mse_loss=loss.item(),
            )

    steps = max(
        steps,
        1,
    )

    return {
        "mse_loss": reduce_mean(
            loss_sum / steps
        ).item(),
    }


def main():
    args = parse_args()

    (
        device,
        local_rank,
        rank,
        distributed,
    ) = setup_distributed(
        args.device
    )

    seed_everything(
        args.seed,
        rank,
    )

    if is_main_process():
        print(
            f"Device: {device}"
        )

        if distributed:
            print(
                "DDP world size:",
                torch.distributed.get_world_size(),
            )

    if args.dry_run:
        train_loader = DummyDataLoader(
            args.batch_size,
            10,
        )

        train_sampler = None

        args.epochs = 2
        args.sample_every = 2

    else:
        from data.dataset import (
            get_dataloaders,
        )

        (
            train_loader,
            _,
            train_sampler,
            _,
        ) = get_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_frac=args.val_frac,
            pin_memory=args.pin_memory,
            distributed=distributed,
            seed=args.seed,
        )

    base_unet = UNet(
        img_channels=3,
        base_channels=args.base_channels,
        time_emb_dim=args.time_emb_dim,
        dropout=args.dropout,
    )

    base_unet = base_unet.to(
        device
    )

    model = wrap_ddp(
        base_unet,
        device,
        local_rank,
        distributed,
    )

    ddpm = DDPM(
        model=model,
        timesteps=args.timesteps,
        schedule=args.schedule,
        device=str(device),
    )

    if is_main_process():
        print_model_info(
            model,
            (
                "DDPM UNet "
                f"(base_channels="
                f"{args.base_channels}, "
                f"T={args.timesteps})"
            ),
        )

    ema = None

    if args.ema_decay > 0:
        ema = EMA(
            unwrap_model(model),
            decay=args.ema_decay,
        )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = get_scheduler(
        optimizer,
        args,
    )

    start_epoch = 1
    best_loss = float("inf")

    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            device=str(device),
        )

        unwrap_model(
            model
        ).load_state_dict(
            checkpoint["model"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        if "scheduler" in checkpoint:
            scheduler.load_state_dict(
                checkpoint["scheduler"]
            )

        if (
            ema is not None
            and "ema" in checkpoint
        ):
            ema.load_state_dict(
                checkpoint["ema"]
            )

        start_epoch = (
            checkpoint.get(
                "epoch",
                0,
            )
            + 1
        )

        best_loss = checkpoint.get(
            "best_loss",
            float("inf"),
        )

    logger = None
    visualizer = None

    if is_main_process():
        logger = TrainingLogger(
            args.output_dir,
            model_name="DDPM",
        )

        visualizer = Visualizer(
            args.output_dir,
            model_name="DDPM",
        )

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        if (
            distributed
            and train_sampler is not None
        ):
            train_sampler.set_epoch(
                epoch
            )

        metrics = train_epoch(
            ddpm,
            train_loader,
            optimizer,
            device,
            args,
            logger,
            epoch,
            ema,
        )

        scheduler.step()

        if not is_main_process():
            continue

        logger.log_epoch(
            epoch=epoch,
            **metrics,
        )

        logger.print_epoch_summary(
            epoch=epoch,
            **metrics,
        )

        is_best = (
            metrics["mse_loss"]
            < best_loss
        )

        if is_best:
            best_loss = (
                metrics["mse_loss"]
            )

        if (
            epoch % args.sample_every == 0
            or epoch == args.epochs
        ):
            if ema is not None:
                sample_model = ema.shadow
            else:
                sample_model = unwrap_model(
                    model
                )

            sample_model.eval()

            sample_ddpm = DDPM(
                model=sample_model,
                timesteps=args.timesteps,
                schedule=args.schedule,
                device=str(device),
            )

            samples = sample_ddpm.sample(
                n=args.n_samples,
                img_shape=(
                    3,
                    128,
                    128,
                ),
                verbose=True,
            )

            save_sample_grid(
                samples,
                (
                    f"{args.output_dir}/"
                    f"samples/"
                    f"gen_ep{epoch:03d}.png"
                ),
                nrow=4,
                title=(
                    f"DDPM Generated — "
                    f"Epoch {epoch}"
                ),
            )

        if (
            epoch % args.save_every == 0
            or epoch == args.epochs
        ):
            state = {
                "epoch": epoch,
                "model": unwrap_model(
                    model
                ).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_loss": best_loss,
                "args": vars(args),
            }

            if ema is not None:
                state["ema"] = (
                    ema.state_dict()
                )

            save_checkpoint(
                state,
                args.output_dir,
                filename=(
                    f"checkpoint_ep"
                    f"{epoch:03d}.pt"
                ),
                is_best=is_best,
            )

            visualizer.plot_curves(
                logger.epoch_history,
                epoch=epoch,
                save=True,
            )

    if is_main_process():
        logger.close()

    cleanup_distributed()


if __name__ == "__main__":
    main()