"""
Обучение DCGAN.

Single GPU:
    python GAN/train.py --config configs/gan.yaml

2x T4:
    torchrun --standalone \
        --nproc_per_node=2 \
        GAN/train.py \
        --config configs/gan.yaml
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset import get_dataloaders
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
    get_config_value,
    load_checkpoint,
    load_config,
    print_model_info,
    save_checkpoint,
    save_sample_grid,
)


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
        "latent_dim": 100,
        "ngf": 64,
        "ndf": 64,
        "epochs": 100,
        "batch_size": 32,
        "lr_g": 2e-4,
        "lr_d": 2e-4,
        "beta1": 0.5,
        "beta2": 0.999,
        "label_smooth": 0.1,
        "n_critic": 1,
        "grad_clip": 0.0,
        "val_frac": 0.0,
        "num_workers": 4,
        "pin_memory": True,
        "save_every": 10,
        "sample_every": 5,
        "log_every": 20,
        "n_samples": 64,
        "output_dir": "checkpoints/gan",
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


def train_epoch(
    generator,
    discriminator,
    loader,
    optimizer_g,
    optimizer_d,
    criterion,
    scaler_g,
    scaler_d,
    device,
    args,
    logger,
    epoch,
    amp_enabled,
):
    generator.train()
    discriminator.train()

    d_sum = torch.zeros(
        (),
        device=device,
    )

    g_sum = torch.zeros(
        (),
        device=device,
    )

    steps = 0

    for step, real in enumerate(
        loader
    ):
        real = real.to(
            device,
            non_blocking=True,
        )

        batch_size = real.shape[0]

        for _ in range(
            max(1, args.n_critic)
        ):
            optimizer_d.zero_grad(
                set_to_none=True
            )

            z = torch.randn(
                batch_size,
                args.latent_dim,
                1,
                1,
                device=device,
            )

            real_labels = smooth_labels(
                batch_size,
                real=True,
                smooth=args.label_smooth,
                device=str(device),
            )

            fake_labels = smooth_labels(
                batch_size,
                real=False,
                smooth=args.label_smooth,
                device=str(device),
            )

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                fake = generator(
                    z
                ).detach()

                d_real = discriminator(
                    real
                )

                d_fake = discriminator(
                    fake
                )

                loss_d = (
                    criterion(
                        d_real,
                        real_labels,
                    )
                    + criterion(
                        d_fake,
                        fake_labels,
                    )
                ) * 0.5

            scaler_d.scale(
                loss_d
            ).backward()

            if args.grad_clip > 0:
                scaler_d.unscale_(
                    optimizer_d
                )

                torch.nn.utils.clip_grad_norm_(
                    discriminator.parameters(),
                    args.grad_clip,
                )

            scaler_d.step(
                optimizer_d
            )

            scaler_d.update()

        optimizer_g.zero_grad(
            set_to_none=True
        )

        z = torch.randn(
            batch_size,
            args.latent_dim,
            1,
            1,
            device=device,
        )

        real_labels = torch.ones(
            batch_size,
            device=device,
        )

        for parameter in discriminator.parameters():
            parameter.requires_grad_(False)

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            fake = generator(z)

            d_fake_for_g = discriminator(
                fake
            )

            loss_g = criterion(
                d_fake_for_g,
                real_labels,
            )

        scaler_g.scale(
            loss_g
        ).backward()

        if args.grad_clip > 0:
            scaler_g.unscale_(
                optimizer_g
            )

            torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                args.grad_clip,
            )

        scaler_g.step(
            optimizer_g
        )

        scaler_g.update()

        for parameter in discriminator.parameters():
            parameter.requires_grad_(True)

        d_sum += loss_d.detach()
        g_sum += loss_g.detach()

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
                d_loss=loss_d.item(),
                g_loss=loss_g.item(),
            )

    return {
        "d_loss": reduce_mean(
            d_sum
            / max(steps, 1)
        ).item(),
        "g_loss": reduce_mean(
            g_sum
            / max(steps, 1)
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

    if args.dry_run:
        args.epochs = 2
        args.in_memory = False

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

    amp_enabled = (
        device.type == "cuda"
    )

    scaler_g = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    scaler_d = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    start_epoch = 1

    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            device=device,
        )

        unwrap_model(
            generator
        ).load_state_dict(
            checkpoint["generator"]
        )

        unwrap_model(
            discriminator
        ).load_state_dict(
            checkpoint["discriminator"]
        )

        if "optimizer_g" in checkpoint:
            optimizer_g.load_state_dict(
                checkpoint["optimizer_g"]
            )

        if "optimizer_d" in checkpoint:
            optimizer_d.load_state_dict(
                checkpoint["optimizer_d"]
            )

        if "scaler_g" in checkpoint:
            scaler_g.load_state_dict(
                checkpoint["scaler_g"]
            )

        if "scaler_d" in checkpoint:
            scaler_d.load_state_dict(
                checkpoint["scaler_d"]
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
            "GAN",
        )

        visualizer = Visualizer(
            args.output_dir,
            "GAN",
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
            scaler_g,
            scaler_d,
            device,
            args,
            logger,
            epoch,
            amp_enabled,
        )

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
                Path(args.output_dir)
                / "samples"
                / f"epoch_{epoch:03d}.png",
                nrow=8,
            )

            generator.train()

        if (
            epoch % args.save_every == 0
            or epoch == args.epochs
        ):
            save_checkpoint(
                {
                    "epoch": epoch,
                    "generator": unwrap_model(
                        generator
                    ).state_dict(),
                    "discriminator": unwrap_model(
                        discriminator
                    ).state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "scaler_g": scaler_g.state_dict(),
                    "scaler_d": scaler_d.state_dict(),
                    "config": args.config_data,
                    "args": vars(args),
                },
                output_dir=args.output_dir,
                filename=(
                    f"checkpoint_ep"
                    f"{epoch:03d}.pt"
                ),
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
