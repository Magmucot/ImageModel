"""
Обучение DCGAN.

Single GPU:

    python GAN/train.py \
        --config configs/gan.yaml

Multi GPU:

    torchrun --standalone \
        --nproc_per_node=4 \
        GAN/train.py \
        --config configs/gan.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from GAN.gan import (
    Discriminator,
    Generator,
    smooth_labels,
    weights_init,
)
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
        default="configs/gan.yaml",
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
        "--ngf",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--ndf",
        type=int,
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
        "--lr_g",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--lr_d",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--beta1",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--beta2",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--label_smooth",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--n_critic",
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
        "grad_clip": 0.0,
        "data_root": "./data",
        "val_frac": 0.05,
        "num_workers": 4,
        "pin_memory": True,
        "latent_dim": 100,
        "ngf": 64,
        "ndf": 64,
        "epochs": 500,
        "batch_size": 32,
        "lr_g": 2e-4,
        "lr_d": 2e-4,
        "beta1": 0.5,
        "beta2": 0.999,
        "label_smooth": 0.1,
        "n_critic": 1,
        "output_dir": "checkpoints/gan",
        "save_every": 10,
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


def set_requires_grad(
    model: torch.nn.Module,
    value: bool,
):
    for parameter in model.parameters():
        parameter.requires_grad_(value)


def train_epoch(
    generator,
    discriminator,
    loader,
    optimizer_g,
    optimizer_d,
    criterion,
    device,
    args,
    logger,
    epoch,
):
    generator.train()
    discriminator.train()

    d_loss_sum = 0.0
    g_loss_sum = 0.0
    dx_sum = 0.0
    dg1_sum = 0.0
    dg2_sum = 0.0

    steps = 0

    for step, real in enumerate(loader):
        real = real.to(
            device,
            non_blocking=True,
        )

        batch_size = real.size(0)

        # --------------------------------------------------------------
        # Discriminator
        # --------------------------------------------------------------

        set_requires_grad(
            discriminator,
            True,
        )

        d_step_loss_sum = 0.0
        d_step_dx_sum = 0.0
        d_step_dg1_sum = 0.0

        for _ in range(args.n_critic):
            optimizer_d.zero_grad(
                set_to_none=True
            )

            real_labels = smooth_labels(
                batch_size,
                real=True,
                smooth=args.label_smooth,
                device=str(device),
            )

            real_output = discriminator(
                real
            )

            loss_real = criterion(
                real_output,
                real_labels,
            )

            noise = torch.randn(
                batch_size,
                args.latent_dim,
                1,
                1,
                device=device,
            )

            fake = generator(
                noise
            ).detach()

            fake_labels = smooth_labels(
                batch_size,
                real=False,
                smooth=args.label_smooth,
                device=str(device),
            )

            fake_output = discriminator(
                fake
            )

            loss_fake = criterion(
                fake_output,
                fake_labels,
            )

            d_loss = (
                loss_real + loss_fake
            ) * 0.5

            d_loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    discriminator.parameters(),
                    args.grad_clip,
                )

            optimizer_d.step()

            d_step_loss_sum += d_loss.detach()
            d_step_dx_sum += real_output.detach().mean()
            d_step_dg1_sum += fake_output.detach().mean()

        n_critic = max(
            args.n_critic,
            1,
        )

        d_loss = (
            d_step_loss_sum
            / n_critic
        )

        real_output_mean = (
            d_step_dx_sum
            / n_critic
        )

        fake_output_mean = (
            d_step_dg1_sum
            / n_critic
        )

        # --------------------------------------------------------------
        # Generator
        # --------------------------------------------------------------

        set_requires_grad(
            discriminator,
            False,
        )

        optimizer_g.zero_grad(
            set_to_none=True
        )

        noise = torch.randn(
            batch_size,
            args.latent_dim,
            1,
            1,
            device=device,
        )

        fake = generator(noise)

        labels = torch.ones(
            batch_size,
            device=device,
        )

        fake_output_for_g = discriminator(
            fake
        )

        g_loss = criterion(
            fake_output_for_g,
            labels,
        )

        g_loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                args.grad_clip,
            )

        optimizer_g.step()

        set_requires_grad(
            discriminator,
            True,
        )

        d_loss_sum += d_loss
        g_loss_sum += g_loss.detach()
        dx_sum += real_output_mean
        dg1_sum += fake_output_mean
        dg2_sum += (
            fake_output_for_g.detach().mean()
        )

        steps += 1

        if (
            is_main_process()
            and (step + 1) % args.log_every == 0
        ):
            logger.log(
                epoch=epoch,
                step=step + 1,
                d_loss=d_loss.item(),
                g_loss=g_loss.item(),
                D_x=real_output_mean.item(),
                D_G_z1=fake_output_mean.item(),
                D_G_z2=fake_output_for_g.mean().item(),
            )

    steps = max(
        steps,
        1,
    )

    return {
        "d_loss": reduce_mean(
            d_loss_sum / steps
        ).item(),
        "g_loss": reduce_mean(
            g_loss_sum / steps
        ).item(),
        "D_x": reduce_mean(
            dx_sum / steps
        ).item(),
        "D_G_z1": reduce_mean(
            dg1_sum / steps
        ).item(),
        "D_G_z2": reduce_mean(
            dg2_sum / steps
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

    generator = Generator(
        latent_dim=args.latent_dim,
        ngf=args.ngf,
    )

    discriminator = Discriminator(
        nc=3,
        ndf=args.ndf,
    )

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    generator = wrap_ddp(
        generator,
        device,
        local_rank,
        distributed,
    )

    discriminator = wrap_ddp(
        discriminator,
        device,
        local_rank,
        distributed,
    )

    if is_main_process():
        print_model_info(
            generator,
            "Generator",
        )

        print_model_info(
            discriminator,
            "Discriminator",
        )

    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=args.lr_g,
        betas=(
            args.beta1,
            args.beta2,
        ),
    )

    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=args.lr_d,
        betas=(
            args.beta1,
            args.beta2,
        ),
    )

    criterion = nn.BCELoss()

    start_epoch = 1

    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            device=str(device),
        )

        unwrap_model(
            generator
        ).load_state_dict(
            checkpoint["G"]
        )

        unwrap_model(
            discriminator
        ).load_state_dict(
            checkpoint["D"]
        )

        optimizer_g.load_state_dict(
            checkpoint["opt_g"]
        )

        optimizer_d.load_state_dict(
            checkpoint["opt_d"]
        )

        start_epoch = (
            checkpoint.get(
                "epoch",
                0,
            )
            + 1
        )

    logger = None
    visualizer = None

    if is_main_process():
        logger = TrainingLogger(
            args.output_dir,
            model_name="GAN",
        )

        visualizer = Visualizer(
            args.output_dir,
            model_name="DCGAN",
        )

    fixed_noise = torch.randn(
        args.n_samples,
        args.latent_dim,
        1,
        1,
        device=device,
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
            generator,
            discriminator,
            train_loader,
            optimizer_g,
            optimizer_d,
            criterion,
            device,
            args,
            logger,
            epoch,
        )

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

        if (
            epoch % args.sample_every == 0
            or epoch == args.epochs
        ):
            generator.eval()

            with torch.no_grad():
                samples = generator(
                    fixed_noise
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
                    f"DCGAN Generated — "
                    f"Epoch {epoch}"
                ),
            )

            generator.train()

        if (
            epoch % args.save_every == 0
            or epoch == args.epochs
        ):
            save_checkpoint(
                {
                    "epoch": epoch,
                    "G": unwrap_model(
                        generator
                    ).state_dict(),
                    "D": unwrap_model(
                        discriminator
                    ).state_dict(),
                    "opt_g": optimizer_g.state_dict(),
                    "opt_d": optimizer_d.state_dict(),
                    "args": vars(args),
                },
                args.output_dir,
                filename=(
                    f"checkpoint_ep"
                    f"{epoch:03d}.pt"
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