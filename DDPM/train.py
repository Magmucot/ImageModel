"""
Обучение DDPM.

Single GPU:
    python DDPM/train.py --config configs/ddpm.yaml

2x T4:
    torchrun --standalone \
        --nproc_per_node=2 \
        DDPM/train.py \
        --config configs/ddpm.yaml
"""

import argparse
import copy
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
    get_config_value,
    load_checkpoint,
    load_config,
    print_model_info,
    save_checkpoint,
    save_sample_grid,
)


class EMA:
    def __init__(
        self,
        model,
        decay=0.9999,
    ):
        if not 0.0 < decay < 1.0:
            raise ValueError(
                "EMA decay должен быть между 0 и 1."
            )

        self.decay = decay
        self.shadow = copy.deepcopy(
            model
        ).eval()

        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(
        self,
        model,
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
        "base_channels": 64,
        "time_emb_dim": 256,
        "timesteps": 1000,
        "schedule": "cosine",
        "dropout": 0.1,
        "epochs": 200,
        "batch_size": 16,
        "lr": 2e-4,
        "weight_decay": 0.0,
        "warmup_epochs": 5,
        "grad_clip": 1.0,
        "ema_decay": 0.9999,
        "val_frac": 0.0,
        "num_workers": 4,
        "pin_memory": True,
        "save_every": 10,
        "sample_every": 20,
        "log_every": 10,
        "n_samples": 16,
        "image_size": 128,
        "in_memory": False,
        "sampling_steps": 50,
        "sampling_eta": 0.0,
        "output_dir": "checkpoints/ddpm",
        "data_root": "./data",
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
    ddpm,
    loader,
    optimizer,
    scaler,
    device,
    args,
    logger,
    epoch,
    ema,
    amp_enabled,
):
    ddpm.model.train()

    loss_sum = torch.zeros(
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
            loss = ddpm.loss_fn(
                images
            )

        scaler.scale(
            loss
        ).backward()

        if args.grad_clip > 0:
            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                ddpm.model.parameters(),
                args.grad_clip,
            )

        scaler.step(
            optimizer
        )

        scaler.update()

        ema.update(
            unwrap_model(
                ddpm.model
            )
        )

        loss_sum += loss.detach()
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
                mse_loss=loss.item(),
            )

    return {
        "mse_loss": reduce_mean(
            loss_sum
            / max(
                steps,
                1,
            )
        ).item()
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
        args.sample_every = 2
        args.save_every = 2

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
        in_memory=args.in_memory,
        image_size=args.image_size,
    )

    base_model = UNet(
        img_channels=3,
        base_channels=args.base_channels,
        time_emb_dim=args.time_emb_dim,
        dropout=args.dropout,
    )

    model = wrap_ddp(
        base_model,
        device,
        local_rank,
        distributed,
    )

    print_model_info(
        model,
        (
            "DDPM UNet "
            f"(base_channels={args.base_channels}, "
            f"T={args.timesteps})"
        ),
    )

    ddpm = DDPM(
        model=model,
        timesteps=args.timesteps,
        schedule=args.schedule,
        device=device,
    )

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

    amp_enabled = (
        device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    start_epoch = 1
    best_loss = float("inf")

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

        if "ema" in checkpoint:
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
            "DDPM",
        )

        visualizer = Visualizer(
            args.output_dir,
            "DDPM",
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
            scaler,
            device,
            args,
            logger,
            epoch,
            ema,
            amp_enabled,
        )

        scheduler.step()

        if not is_main_process():
            continue

        logger.log_epoch(
            epoch,
            **metrics,
        )

        logger.print_epoch_summary(
            epoch,
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
            sample_model = ema.shadow
            sample_model.eval()

            sample_ddpm = DDPM(
                model=sample_model,
                timesteps=args.timesteps,
                schedule=args.schedule,
                device=device,
            )

            samples = sample_ddpm.ddim_sample(
                n=args.n_samples,
                img_shape=(
                    3,
                    args.image_size,
                    args.image_size,
                ),
                sampling_steps=args.sampling_steps,
                eta=args.sampling_eta,
                verbose=True,
            )

            save_sample_grid(
                samples,
                Path(args.output_dir)
                / "samples"
                / f"epoch_{epoch:03d}.png",
                nrow=4,
                title=(
                    f"DDIM "
                    f"{args.sampling_steps} steps"
                ),
            )

        if (
            epoch % args.save_every == 0
            or epoch == args.epochs
            or is_best
        ):
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": unwrap_model(
                        model
                    ).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "ema": ema.state_dict(),
                    "best_loss": best_loss,
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
