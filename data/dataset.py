"""
Dataset и DataLoader для VAE/GAN/DDPM.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torchvision.io as io
from PIL import Image
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
)
from torchvision import transforms


VALID_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
    }
)


def scan_image_files(
    root: str | Path,
) -> list[str]:
    root = Path(root).expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}"
        )

    paths = []

    for dirpath, _, filenames in os.walk(
        root
    ):
        for filename in filenames:
            extension = (
                Path(filename)
                .suffix
                .lower()
            )

            if extension in VALID_EXTENSIONS:
                paths.append(
                    str(
                        Path(dirpath)
                        / filename
                    )
                )

    paths.sort()

    if not paths:
        raise FileNotFoundError(
            f"Images not found in {root}"
        )

    return paths


def _load_image(
    path: str,
    image_size: int,
) -> torch.Tensor:
    try:
        tensor = io.read_image(
            path,
            mode=io.ImageReadMode.RGB,
        )

        if tensor.ndim != 3:
            raise ValueError(
                f"Invalid image shape: "
                f"{tensor.shape}"
            )

        if tensor.shape[-2:] != (
            image_size,
            image_size,
        ):
            tensor = transforms.functional.resize(
                tensor,
                [
                    image_size,
                    image_size,
                ],
                antialias=True,
            )

        return tensor

    except Exception:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = image.resize(
                (
                    image_size,
                    image_size,
                ),
                Image.Resampling.LANCZOS,
            )

            return torch.from_numpy(
                np.asarray(image)
            ).permute(
                2,
                0,
                1,
            ).contiguous()


def default_train_transform(
    image_size: int = 128,
):
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(
                p=0.5
            ),
            transforms.ConvertImageDtype(
                torch.float32
            ),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ),
        ]
    )


def default_val_transform(
    image_size: int = 128,
):
    return transforms.Compose(
        [
            transforms.ConvertImageDtype(
                torch.float32
            ),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ),
        ]
    )


class FFHQDataset(Dataset):
    def __init__(
        self,
        root: str | Path | None = None,
        paths: Sequence[str] | None = None,
        split: str = "train",
        transform: Callable | None = None,
        val_frac: float = 0.05,
        in_memory: bool = False,
        image_size: int = 128,
        cache: torch.Tensor | None = None,
    ):
        super().__init__()

        if not 0.0 <= val_frac < 1.0:
            raise ValueError(
                "val_frac должен быть в [0, 1)."
            )

        if image_size <= 0:
            raise ValueError(
                "image_size должен быть > 0."
            )

        if paths is None:
            if root is None:
                raise ValueError(
                    "Передайте root или paths."
                )

            paths = scan_image_files(root)

        all_paths = sorted(
            list(paths)
        )

        if not all_paths:
            raise ValueError(
                "Dataset пуст."
            )

        n_total = len(all_paths)

        n_val = (
            max(
                1,
                int(
                    n_total * val_frac
                ),
            )
            if val_frac > 0
            else 0
        )

        n_train = n_total - n_val

        if n_train <= 0:
            raise ValueError(
                "Train split пуст."
            )

        if split == "train":
            self.paths = all_paths[
                :n_train
            ]
        elif split == "val":
            self.paths = all_paths[
                n_train:
            ]
        elif split == "all":
            self.paths = all_paths
        else:
            raise ValueError(
                f"Unknown split: {split}"
            )

        self.image_size = image_size
        self.transform = (
            transform
            if transform is not None
            else default_train_transform(
                image_size
            )
        )

        self.in_memory = in_memory

        self.cache = cache

        if self.in_memory and self.cache is None:
            self.cache = self._build_cache(
                all_paths,
                image_size,
            )

        self._index_offset = 0

        if (
            self.in_memory
            and cache is not None
        ):
            all_path_to_index = {
                path: index
                for index, path in enumerate(
                    all_paths
                )
            }

            self._indices = torch.tensor(
                [
                    all_path_to_index[path]
                    for path in self.paths
                ],
                dtype=torch.long,
            )
        else:
            self._indices = None

    @staticmethod
    def _build_cache(
        paths: Sequence[str],
        image_size: int,
    ) -> torch.Tensor:
        cache = torch.empty(
            (
                len(paths),
                3,
                image_size,
                image_size,
            ),
            dtype=torch.uint8,
        )

        for index, path in enumerate(
            paths
        ):
            cache[index].copy_(
                _load_image(
                    path,
                    image_size,
                )
            )

        return cache

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(
        self,
        index: int,
    ) -> torch.Tensor:
        if (
            self.in_memory
            and self.cache is not None
            and self._indices is not None
        ):
            source_index = int(
                self._indices[index]
            )

            image = self.cache[
                source_index
            ]
        else:
            image = _load_image(
                self.paths[index],
                self.image_size,
            )

        return self.transform(
            image
        )


class ImageDataset(FFHQDataset):
    def __init__(
        self,
        root: str | Path,
        image_size: int = 128,
        transform: Callable | None = None,
        in_memory: bool = False,
        **kwargs,
    ):
        super().__init__(
            root=root,
            split="all",
            transform=(
                transform
                if transform is not None
                else default_train_transform(
                    image_size
                )
            ),
            val_frac=0.0,
            in_memory=in_memory,
            image_size=image_size,
            **kwargs,
        )


def get_dataloaders(
    data_root: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    val_frac: float = 0.05,
    pin_memory: bool = True,
    distributed: bool = False,
    seed: int = 42,
    in_memory: bool = False,
    image_size: int = 128,
):
    all_paths = scan_image_files(
        data_root
    )

    shared_cache = None

    if in_memory:
        shared_cache = (
            FFHQDataset._build_cache(
                all_paths,
                image_size,
            )
        )

    train_ds = FFHQDataset(
        paths=all_paths,
        split="train",
        val_frac=val_frac,
        transform=default_train_transform(
            image_size
        ),
        in_memory=in_memory,
        image_size=image_size,
        cache=shared_cache,
    )

    val_ds = None

    if val_frac > 0:
        val_ds = FFHQDataset(
            paths=all_paths,
            split="val",
            val_frac=val_frac,
            transform=default_val_transform(
                image_size
            ),
            in_memory=in_memory,
            image_size=image_size,
            cache=shared_cache,
        )

    train_sampler = None
    val_sampler = None

    if distributed:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )

        if val_ds is not None:
            val_sampler = DistributedSampler(
                val_ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=seed,
                drop_last=False,
            )

    effective_workers = (
        0
        if in_memory
        else max(
            0,
            num_workers,
        )
    )

    use_pin_memory = (
        pin_memory
        and torch.cuda.is_available()
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(
            train_sampler is None
        ),
        sampler=train_sampler,
        num_workers=effective_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(
            effective_workers > 0
        ),
        drop_last=True,
    )

    val_loader = None

    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=effective_workers,
            pin_memory=use_pin_memory,
            persistent_workers=(
                effective_workers > 0
            ),
            drop_last=False,
        )

    return (
        train_loader,
        val_loader,
        train_sampler,
        val_sampler,
    )
