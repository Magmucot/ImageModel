"""
Обучение beta-VAE.

Single GPU:

    python VAE/train.py \
        --config configs/vae.yaml

Multi GPU:

    torchrun --standalone \
        --nproc_per_node=4 \
        VAE/train.py \
        --config configs/vae.yaml
"""

from __future__ import annotations

import argparse
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

from VAE.vae import VAE, vae_loss
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
        default="configs/vae.yaml",
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
        "--latent_dim",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--beta",
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
        "latent_dim": 256,
        "beta": 4.0,
        "epochs": 50,
        "batch_size": 32,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "warmup_epochs": 2,
        "output_dir": "checkpoints/vae",
        "save_every": 5,
        "sample_every": 5,
        "log_every": 20,
        "n_samples": 64,
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
        batch_size: int,
        n_batches: int,
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
    model,
    loader,
    optimizer,
    device,
    args,
    logger,
    epoch,
):
    model.train()

    total_sum = 0.0
    recon_sum = 0.0
    kl_sum = 0.0

    steps = 0

    base_model = unwrap_model(model)

    for step, batch in enumerate(loader):
        images = batch.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        recon, mu, logvar = model(
            images
        )

        total, recon_loss, kl_loss = (
            vae_loss(
                recon,
                images,
                mu,
                logvar,
                beta=base_model.beta,
            )
        )

        total.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )

        optimizer.step()

        total_sum += total.detach()
        recon_sum += recon_loss.detach()
        kl_sum += kl_loss.detach()

        steps += 1

        if (
            is_main_process()
            and (step + 1) % args.log_every == 0
        ):
            logger.log(
                epoch=epoch,
                step=step + 1,
                total_loss=total.item(),
                recon_loss=recon_loss.item(),
                kl_loss=kl_loss.item(),
            )

    steps = max(steps, 1)

    return {
        "total_loss": reduce_mean(
            total_sum / steps
        ).item(),
        "recon_loss": reduce_mean(
            recon_sum / steps
        ).item(),
        "kl_loss": reduce_mean(
            kl_sum / steps
        ).item(),
    }


@torch.no_grad()
def val_epoch(
    model,
    loader,
    device,
):
    model.eval()

    base_model = unwrap_model(model)

    total_sum = 0.0
    recon_sum = 0.0
    kl_sum = 0.0
    steps = 0

    for batch in loader:
        images = batch.to(
            device,
            non_blocking=True,
        )

        recon, mu, logvar = model(
            images
        )

        total, recon_loss, kl_loss = (
            vae_loss(
                recon,
                images,
                mu,
                logvar,
                beta=base_model.beta,
            )
        )

        total_sum += total.detach()
        recon_sum += recon_loss.detach()
        kl_sum += kl_loss.detach()

        steps += 1

    steps = max(steps, 1)

    return {
        "val_total": reduce_mean(
            total_sum / steps
        ).item(),
        "val_recon": reduce_mean(
            recon_sum / steps
        ).item(),
        "val_kl": reduce_mean(
            kl_sum / steps
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

    # ВАЖНО:
    # rank существует только после setup_distributed().
    #
    # Для DDP каждый процесс получает отдельную RNG sequence:
    # rank 0 -> seed
    # rank 1 -> seed + 1
    # rank 2 -> seed + 2
    # ...
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
            5,
        )

        val_loader = DummyDataLoader(
            args.batch_size,
            2,
        )

        train_sampler = None

        args.epochs = 2

    else:
        from data.dataset import (
            get_dataloaders,
        )

        (
            train_loader,
            val_loader,
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

    model = VAE(
        latent_dim=args.latent_dim,
        beta=args.beta,
    )

    model = wrap_ddp(
        model,
        device,
        local_rank,
        distributed,
    )

    if is_main_process():
        print_model_info(
            model,
            (
                "β-VAE "
                f"(latent_dim={args.latent_dim}, "
                f"β={args.beta})"
            ),
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
    best_val = float("inf")

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

        start_epoch = (
            checkpoint.get(
                "epoch",
                0,
            )
            + 1
        )

        best_val = checkpoint.get(
            "best_val",
            float("inf"),
        )

    logger = None
    visualizer = None

    if is_main_process():
        logger = TrainingLogger(
            args.output_dir,
            model_name="VAE",
        )

        visualizer = Visualizer(
            args.output_dir,
            model_name="β-VAE",
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

        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args,
            logger,
            epoch,
        )

        val_metrics = val_epoch(
            model,
            val_loader,
            device,
        )

        scheduler.step()

        if not is_main_process():
            continue

        metrics = {
            **train_metrics,
            **val_metrics,
        }

        logger.log_epoch(
            epoch=epoch,
            **metrics,
        )

        logger.print_epoch_summary(
            epoch=epoch,
            **metrics,
        )

        is_best = (
            val_metrics["val_total"]
            < best_val
        )

        if is_best:
            best_val = (
                val_metrics["val_total"]
            )

        if (
            epoch % args.save_every == 0
            or epoch == args.epochs
        ):
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": unwrap_model(
                        model
                    ).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val": best_val,
                    "args": vars(args),
                },
                args.output_dir,
                filename=(
                    f"checkpoint_ep"
                    f"{epoch:03d}.pt"
                ),
                is_best=is_best,
            )

        if (
            epoch % args.sample_every == 0
            or epoch == args.epochs
        ):
            samples = unwrap_model(
                model
            ).sample(
                n=args.n_samples,
                device=str(device),
            )

            save_sample_grid(
                samples,
                (
                    f"{args.output_dir}/"
                    f"samples/"
                    f"gen_ep{epoch:03d}.png"
                ),
                nrow=8,
                title=(
                    f"β-VAE Generated — "
                    f"Epoch {epoch}"
                ),
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