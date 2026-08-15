"""
Обучение DDPM.

Single GPU:
    python DDPM/train.py --config configs/ddpm.yaml

<<<<<<< HEAD
2x T4:
||||||| parent of 24ef89e (more fixes)
    python DDPM/train.py \
        --config configs/ddpm.yaml

Multi GPU:

=======
    python DDPM/train.py \
        --config configs/ddpm.yaml

2x T4:

>>>>>>> 24ef89e (more fixes)
    torchrun --standalone \
        --nproc_per_node=2 \
        DDPM/train.py \
        --config configs/ddpm.yaml

Можно переопределить путь к датасету:

    torchrun --standalone \
        --nproc_per_node=2 \
        DDPM/train.py \
        --config configs/ddpm.yaml \
        --data_root /path/to/ffhq
"""

import argparse
import copy
<<<<<<< HEAD
import sys
||||||| parent of 24ef89e (more fixes)
import os
import sys
=======
>>>>>>> 24ef89e (more fixes)
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

<<<<<<< HEAD
from data.dataset import get_dataloaders
||||||| parent of 24ef89e (more fixes)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

=======
import sys

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import get_dataloaders
>>>>>>> 24ef89e (more fixes)
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
<<<<<<< HEAD
||||||| parent of 24ef89e (more fixes)
    """EMA настоящей UNet-модели."""

=======
    """Exponential Moving Average модели."""

>>>>>>> 24ef89e (more fixes)
    def __init__(
        self,
        model,
        decay=0.9999,
    ):
<<<<<<< HEAD
        if not 0.0 < decay < 1.0:
            raise ValueError(
                "EMA decay должен быть между 0 и 1."
            )
||||||| parent of 24ef89e (more fixes)
        self.decay = decay
=======
        if not 0.0 < decay < 1.0:
            raise ValueError(
                "EMA decay должен быть "
                "между 0 и 1."
            )

        self.decay = decay
>>>>>>> 24ef89e (more fixes)

        self.decay = decay
        self.shadow = copy.deepcopy(
            model
        ).eval()

        for parameter in (
            self.shadow.parameters()
        ):
            parameter.requires_grad_(
                False
            )

    @torch.no_grad()
    def update(
        self,
<<<<<<< HEAD
        model,
    ):
||||||| parent of 24ef89e (more fixes)
        model: torch.nn.Module,
    ):
=======
        model: torch.nn.Module,
    ) -> None:
>>>>>>> 24ef89e (more fixes)
        for shadow, current in zip(
            self.shadow.parameters(),
            model.parameters(),
        ):
            shadow.mul_(
                self.decay
            ).add_(
                current,
                alpha=(
                    1.0
                    - self.decay
                ),
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
    ) -> None:
        self.shadow.load_state_dict(
            state_dict
        )


<<<<<<< HEAD
||||||| parent of 24ef89e (more fixes)
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


=======
class DummyDataLoader:
    """Минимальный loader для --dry_run."""

    def __init__(
        self,
        batch_size: int,
        n_batches: int = 4,
        image_size: int = 128,
    ):
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.image_size = image_size

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(
            self.n_batches
        ):
            yield torch.randn(
                self.batch_size,
                3,
                self.image_size,
                self.image_size,
            )


>>>>>>> 24ef89e (more fixes)
def parse_args():
    parser = argparse.ArgumentParser(
        description="DDPM training"
    )

    parser.add_argument(
        "--config",
        default="configs/ddpm.yaml",
    )

    parser.add_argument(
        "--data_root",
        default=None,
    )

    parser.add_argument(
<<<<<<< HEAD
||||||| parent of 24ef89e (more fixes)
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
=======
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
        type=lambda value: (
            value.lower() == "true"
        ),
        default=None,
    )

    parser.add_argument(
        "--in_memory",
        type=lambda value: (
            value.lower() == "true"
        ),
        default=None,
        help=(
            "Cache the full dataset in each process. "
            "Keep false for multi-process DDP."
        ),
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
        "--image_size",
        type=int,
        default=None,
    )

    parser.add_argument(
>>>>>>> 24ef89e (more fixes)
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
<<<<<<< HEAD
||||||| parent of 24ef89e (more fixes)
        "data_root": "./data",
        "val_frac": 0.05,
        "num_workers": 4,
        "pin_memory": True,
=======
        "data_root": "./data",
        "val_frac": 0.05,
        "num_workers": 4,
        "pin_memory": True,
        "in_memory": False,
>>>>>>> 24ef89e (more fixes)
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
<<<<<<< HEAD
        "val_frac": 0.0,
        "num_workers": 4,
        "pin_memory": True,
||||||| parent of 24ef89e (more fixes)
        "output_dir": "checkpoints/ddpm",
=======
        "output_dir": (
            "checkpoints/ddpm"
        ),
>>>>>>> 24ef89e (more fixes)
        "save_every": 10,
        "sample_every": 20,
        "log_every": 10,
        "n_samples": 16,
<<<<<<< HEAD
        "image_size": 128,
        "in_memory": False,
        "sampling_steps": 50,
        "sampling_eta": 0.0,
        "output_dir": "checkpoints/ddpm",
        "data_root": "./data",
||||||| parent of 24ef89e (more fixes)
=======
        "image_size": 128,
>>>>>>> 24ef89e (more fixes)
        "seed": 42,
    }

    for key, default in defaults.items():
<<<<<<< HEAD
        setattr(
            args,
            key,
            get_config_value(
                config,
||||||| parent of 24ef89e (more fixes)
        if getattr(args, key) is None:
            setattr(
                args,
=======
        if getattr(
            args,
            key,
        ) is None:
            setattr(
                args,
>>>>>>> 24ef89e (more fixes)
                key,
                default,
            ),
        )

    if args.data_root is None:
        args.data_root = "./data"

    args.config_data = config

    # Явный CLI override имеет приоритет.
    args.config_data = config

    return args


def get_scheduler(
    optimizer,
    args,
):
<<<<<<< HEAD
    if args.warmup_epochs <= 0:
        return CosineAnnealingLR(
||||||| parent of 24ef89e (more fixes)
    if args.warmup_epochs > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=args.warmup_epochs,
        )

        cosine = CosineAnnealingLR(
=======
    """Warmup + cosine."""

    if args.warmup_epochs > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=args.warmup_epochs,
        )

        cosine = CosineAnnealingLR(
>>>>>>> 24ef89e (more fixes)
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


def should_sample(
    epoch: int,
    epochs: int,
    sample_every: int,
    dry_run: bool,
) -> bool:
    """Dry run validates training without slow diffusion sampling."""
    if dry_run:
        return False

    return (
        epoch % sample_every == 0
        or epoch == epochs
    )


def train_epoch(
    ddpm: DDPM,
    loader,
    optimizer,
    scaler,
    device,
    args,
    logger,
<<<<<<< HEAD
    epoch,
    ema,
    amp_enabled,
||||||| parent of 24ef89e (more fixes)
    epoch,
    ema,
=======
    epoch: int,
    ema: EMA | None,
>>>>>>> 24ef89e (more fixes)
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

<<<<<<< HEAD
        scaler.step(
            optimizer
        )
||||||| parent of 24ef89e (more fixes)
        optimizer.step()
=======
        scaler.step(
            optimizer
        )

        scaler.update()
>>>>>>> 24ef89e (more fixes)

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
<<<<<<< HEAD
            loss_sum
            / max(
                steps,
                1,
            )
        ).item()
||||||| parent of 24ef89e (more fixes)
            loss_sum / steps
        ).item(),
=======
            loss_sum / steps
        ).item()
>>>>>>> 24ef89e (more fixes)
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
<<<<<<< HEAD
||||||| parent of 24ef89e (more fixes)
        train_loader = DummyDataLoader(
            args.batch_size,
            10,
        )

        train_sampler = None

=======
        train_loader = DummyDataLoader(
            args.batch_size,
            n_batches=2,
            image_size=args.image_size,
        )

        train_sampler = None

>>>>>>> 24ef89e (more fixes)
        args.epochs = 2
        args.in_memory = False
        args.sample_every = 2
        args.save_every = 2

<<<<<<< HEAD
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
||||||| parent of 24ef89e (more fixes)
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
=======
    else:
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

    # ВАЖНО:
    # сначала создаётся обычный UNet.
    # Только после .to(device) он передаётся в DDP.
    base_unet = UNet(
>>>>>>> 24ef89e (more fixes)
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

<<<<<<< HEAD
    amp_enabled = (
        device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

||||||| parent of 24ef89e (more fixes)
=======
    # T4 отлично подходит для FP16.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type == "cuda"
        ),
    )

>>>>>>> 24ef89e (more fixes)
    start_epoch = 1
    best_loss = float("inf")

    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            device=device,
        )

        model_state = checkpoint.get(
            "model"
        )

        if model_state is None:
            model_state = checkpoint.get(
                "model_state_dict"
            )

        if model_state is None:
            raise KeyError(
                "В checkpoint отсутствует "
                "'model' или "
                "'model_state_dict'."
            )

        unwrap_model(
            model
        ).load_state_dict(
            model_state
        )

<<<<<<< HEAD
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer"]
            )
||||||| parent of 24ef89e (more fixes)
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )
=======
        optimizer_state = (
            checkpoint.get(
                "optimizer"
            )
            or checkpoint.get(
                "optimizer_state_dict"
            )
        )
>>>>>>> 24ef89e (more fixes)

        if optimizer_state:
            optimizer.load_state_dict(
                optimizer_state
            )

        scheduler_state = (
            checkpoint.get(
                "scheduler"
            )
            or checkpoint.get(
                "scheduler_state_dict"
            )
        )

        if scheduler_state:
            scheduler.load_state_dict(
                scheduler_state
            )

        scaler_state = (
            checkpoint.get(
                "scaler"
            )
            or checkpoint.get(
                "scaler_state_dict"
            )
        )

        if scaler_state:
            scaler.load_state_dict(
                scaler_state
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

        if should_sample(
            epoch,
            args.epochs,
            args.sample_every,
            args.dry_run,
        ):
<<<<<<< HEAD
            sample_model = ema.shadow
||||||| parent of 24ef89e (more fixes)
            if ema is not None:
                sample_model = ema.shadow
            else:
                sample_model = unwrap_model(
                    model
                )

=======
            sample_model = (
                ema.shadow
                if ema is not None
                else unwrap_model(
                    model
                )
            )

>>>>>>> 24ef89e (more fixes)
            sample_model.eval()

            sample_ddpm = DDPM(
                model=sample_model,
                timesteps=args.timesteps,
                schedule=args.schedule,
                device=device,
            )

<<<<<<< HEAD
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
||||||| parent of 24ef89e (more fixes)
            samples = sample_ddpm.sample(
                n=args.n_samples,
                img_shape=(
                    3,
                    128,
                    128,
                ),
                verbose=True,
            )
=======
            with torch.no_grad():
                samples = (
                    sample_ddpm.sample(
                        n=args.n_samples,
                        img_shape=(
                            3,
                            args.image_size,
                            args.image_size,
                        ),
                        verbose=True,
                    )
                )
>>>>>>> 24ef89e (more fixes)

            save_sample_grid(
                samples,
<<<<<<< HEAD
                Path(args.output_dir)
                / "samples"
                / f"epoch_{epoch:03d}.png",
||||||| parent of 24ef89e (more fixes)
                (
                    f"{args.output_dir}/"
                    f"samples/"
                    f"gen_ep{epoch:03d}.png"
                ),
=======
                (
                    Path(
                        args.output_dir
                    )
                    / "samples"
                    / (
                        f"gen_ep"
                        f"{epoch:03d}.png"
                    )
                ),
>>>>>>> 24ef89e (more fixes)
                nrow=4,
                title=(
<<<<<<< HEAD
                    f"DDIM "
                    f"{args.sampling_steps} steps"
||||||| parent of 24ef89e (more fixes)
                    f"DDPM Generated — "
                    f"Epoch {epoch}"
=======
                    f"DDPM Generated "
                    f"— Epoch {epoch}"
>>>>>>> 24ef89e (more fixes)
                ),
            )

        if (
            epoch % args.save_every == 0
            or epoch == args.epochs
            or is_best
        ):
<<<<<<< HEAD
||||||| parent of 24ef89e (more fixes)
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

=======
            state = {
                "epoch": epoch,
                "model": unwrap_model(
                    model
                ).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_loss": best_loss,
                "args": vars(args),
                "config": args.config_data,
            }

            if ema is not None:
                state["ema"] = (
                    ema.state_dict()
                )

>>>>>>> 24ef89e (more fixes)
            save_checkpoint(
<<<<<<< HEAD
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
||||||| parent of 24ef89e (more fixes)
                state,
                args.output_dir,
=======
                state,
                output_dir=args.output_dir,
>>>>>>> 24ef89e (more fixes)
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
