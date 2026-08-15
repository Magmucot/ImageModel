"""
Обучение VAE.

Single GPU:
    python VAE/train.py --config configs/vae.yaml

2x T4:
    torchrun --standalone \
        --nproc_per_node=2 \
        VAE/train.py \
        --config configs/vae.yaml
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

from data.dataset import get_dataloaders
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
    get_config_value,
    load_checkpoint,
    load_config,
    print_model_info,
    save_checkpoint,
    save_sample_grid,
)
from VAE.vae import VAE, vae_loss


def parse_args():
    parser = argparse.ArgumentParser(
        description="VAE training"
    )

    parser.add_argument(
        "--config",
        default="configs/vae.yaml",
    )

    parser.add_argument(
        "--data_root",
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
        "--dry_run",
        action="store_true",
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    defaults = {
        "latent_dim": 256,
        "beta": 4.0,
        "epochs": 50,
        "batch_size": 32,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "warmup_epochs": 2,
        "grad_clip": 1.0,
        "val_frac": 0.05,
        "num_workers": 4,
        "pin_memory": True,
        "save_every": 5,
        "sample_every": 5,
        "log_every": 20,
        "n_samples": 64,
        "output_dir": "checkpoints/vae",
        "data_root": "./data",
        "image_size": 128,
        "in_memory": False,
        "seed": 42,
    }

    for key, default in defaults.items():
        setattr(
            args,
            key,
            get_config_value(
                config,
                key,
                default,
            ),
        )

    if args.data_root is None:
        args.data_root = "./data"

    args.config_data = config

    return args


def get_scheduler(
    optimizer,
    args,
):
    if args.warmup_epochs <= 0:
        return CosineAnnealingLR(
            optimizer,
            T_max=max(
                1,
                args.epochs,
            ),
            eta_min=args.lr * 0.01,
        )

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


def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    args,
    logger,
    epoch,
    amp_enabled,
):
    model.train()

    total_sum = torch.zeros(
        (),
        device=device,
    )

    recon_sum = torch.zeros(
        (),
        device=device,
    )

    kl_sum = torch.zeros(
        (),
        device=device,
    )

    steps = 0

    for step, images in enumerate(
        loader
    ):
        images = images.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            recon, mu, logvar = model(
                images
            )

            (
                total,
                recon_loss,
                kl_loss,
            ) = vae_loss(
                recon,
                images,
                mu,
                logvar,
                beta=args.beta,
            )

        scaler.scale(
            total
        ).backward()

        if args.grad_clip > 0:
            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )

        scaler.step(
            optimizer
        )

        scaler.update()

        total_sum += total.detach()
        recon_sum += recon_loss.detach()
        kl_sum += kl_loss.detach()

        steps += 1

        if (
            is_main_process()
            and (
                step + 1
            ) % args.log_every
            == 0
        ):
            logger.log(
                epoch=epoch,
                step=step + 1,
                total_loss=total.item(),
                recon_loss=recon_loss.item(),
                kl_loss=kl_loss.item(),
            )

    steps = max(
        steps,
        1,
    )

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
    args,
    amp_enabled,
):
    model.eval()

    if loader is None:
        return {
            "val_total": 0.0,
            "val_recon": 0.0,
            "val_kl": 0.0,
        }

    total_sum = torch.zeros(
        (),
        device=device,
    )

    recon_sum = torch.zeros(
        (),
        device=device,
    )

    kl_sum = torch.zeros(
        (),
        device=device,
    )

    steps = 0

    base_model = unwrap_model(
        model
    )

    for images in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            recon, mu, logvar = model(
                images
            )

            (
                total,
                recon_loss,
                kl_loss,
            ) = vae_loss(
                recon,
                images,
                mu,
                logvar,
                beta=base_model.beta,
            )

        total_sum += total.detach()
        recon_sum += recon_loss.detach()
        kl_sum += kl_loss.detach()

        steps += 1

    steps = max(
        steps,
        1,
    )

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

            print(
                "Global batch size:",
                args.batch_size
                * torch.distributed.get_world_size(),
            )

    if args.dry_run:
        args.epochs = 2
        args.in_memory = False

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
        in_memory=args.in_memory,
        image_size=args.image_size,
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

    print_model_info(
        model,
        "VAE",
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

    amp_enabled = (
        device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    start_epoch = 1
    best_val = float("inf")

    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            device=device,
        )

        unwrap_model(
            model
        ).load_state_dict(
            checkpoint["model"]
        )

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

        if "scheduler" in checkpoint:
            scheduler.load_state_dict(
                checkpoint["scheduler"]
            )

        if "scaler" in checkpoint:
            scaler.load_state_dict(
                checkpoint["scaler"]
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
            "VAE",
        )

        visualizer = Visualizer(
            args.output_dir,
            "VAE",
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
            scaler,
            device,
            args,
            logger,
            epoch,
            amp_enabled,
        )

        val_metrics = val_epoch(
            model,
            val_loader,
            device,
            args,
            amp_enabled,
        )

        scheduler.step()

        if not is_main_process():
            continue

        metrics = {
            **train_metrics,
            **val_metrics,
        }

        logger.log_epoch(
            epoch,
            **metrics,
        )

        logger.print_epoch_summary(
            epoch,
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

        should_sample = (
            epoch % args.sample_every == 0
            or epoch == args.epochs
        )

        should_save = (
            epoch % args.save_every == 0
            or epoch == args.epochs
            or is_best
        )

        if should_sample:
            samples = unwrap_model(
                model
            ).sample(
                n=args.n_samples,
                device=str(device),
            )

            save_sample_grid(
                samples,
                (
                    Path(args.output_dir)
                    / "samples"
                    / f"epoch_{epoch:03d}.png"
                ),
                nrow=8,
            )

        if should_save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": unwrap_model(
                        model
                    ).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val": best_val,
                    "config": args.config_data,
                    "args": vars(args),
                },
                output_dir=args.output_dir,
                filename=(
                    f"checkpoint_ep"
                    f"{epoch:03d}.pt"
                ),
                is_best=is_best,
            )

            visualizer.plot_curves(
                logger.epoch_history,
                epoch,
                save=True,
            )

    if is_main_process():
        logger.close()

    cleanup_distributed()


if __name__ == "__main__":
    main()
