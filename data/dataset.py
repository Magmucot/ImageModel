"""
Загрузка данных FFHQ для обучения VAE, GAN, DDPM.

Поддерживает single-GPU и DistributedDataParallel.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
)
from torchvision import transforms


class FFHQDataset(Dataset):
    """
    Датасет FFHQ 128x128.

    Ожидаемая структура:

        data/
            thumbnails128x128/
                ...
                *.png
    """

    EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        val_frac: float = 0.05,
    ):
        self.root = Path(root)

        self.transform = (
            transform
            if transform is not None
            else self._default_transform()
        )

        all_paths = self._gather_images()

        if not all_paths:
            raise FileNotFoundError(
                f"Изображения не найдены в {self.root}. "
                "Убедитесь, что FFHQ thumbnails128x128 скачаны."
            )

        n_val = max(
            1,
            int(len(all_paths) * val_frac),
        )

        n_train = len(all_paths) - n_val

        if split == "train":
            self.paths = all_paths[:n_train]

        elif split == "val":
            self.paths = all_paths[n_train:]

        elif split == "all":
            self.paths = all_paths

        else:
            raise ValueError(
                f"Неизвестный split: {split!r}"
            )

        print(
            f"[Dataset] split={split!r}, "
            f"images={len(self.paths):,}"
        )

    def _gather_images(self) -> list[str]:
        """Рекурсивно собирает изображения."""

        paths = []

        for extension in self.EXTENSIONS:
            paths.extend(
                self.root.rglob(
                    f"*{extension}"
                )
            )

            paths.extend(
                self.root.rglob(
                    f"*{extension.upper()}"
                )
            )

        paths = sorted(set(paths))

        return [
            str(path)
            for path in paths
        ]

    @staticmethod
    def _default_transform():
        return transforms.Compose(
            [
                transforms.Resize(
                    (128, 128)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(
        self,
        idx: int,
    ):
        image = Image.open(
            self.paths[idx]
        ).convert("RGB")

        return self.transform(image)


def get_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    val_frac: float = 0.05,
    pin_memory: bool = True,
    distributed: bool = False,
    seed: int = 42,
):
    """
    Создаёт train/validation DataLoader.

    При DDP batch_size является batch size одного GPU.
    """

    train_ds = FFHQDataset(
        data_root,
        split="train",
        val_frac=val_frac,
    )

    val_ds = FFHQDataset(
        data_root,
        split="val",
        val_frac=val_frac,
    )

    val_ds.transform = transforms.Compose(
        [
            transforms.Resize(
                (128, 128)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    train_sampler = None
    val_sampler = None

    if distributed:
        train_sampler = DistributedSampler(
            train_ds,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )

        val_sampler = DistributedSampler(
            val_ds,
            shuffle=False,
            seed=seed,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    return (
        train_loader,
        val_loader,
        train_sampler,
        val_sampler,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
    )

    args = parser.parse_args()

    train_loader, val_loader, _, _ = (
        get_dataloaders(
            args.data_root,
            batch_size=8,
            num_workers=2,
        )
    )

    batch = next(iter(train_loader))

    print(
        f"Батч: {batch.shape}, "
        f"min={batch.min():.3f}, "
        f"max={batch.max():.3f}"
    )

    print(
        f"Train батчей: {len(train_loader)}"
    )

    print(
        f"Val батчей: {len(val_loader)}"
    )