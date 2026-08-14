"""
Загрузка и эффективный пайплайн данных FFHQ для VAE, GAN и DDPM.

Поддерживает Single-GPU и DistributedDataParallel (DDP) с in-memory RAM кэшированием.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
)
import torchvision.io as io
from torchvision import transforms

VALID_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
)


def scan_image_files(root: Path | str) -> list[str]:
    """Однопроходное быстрое сканирование директории без рекурсивного rglob."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Директория с датасетом не найдена: {root_path}")

    paths: list[str] = []
    for dirpath, _, filenames in os.walk(root_path):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in VALID_EXTENSIONS:
                paths.append(os.path.join(dirpath, name))

    if not paths:
        raise FileNotFoundError(
            f"Изображения не найдены в {root_path}. Проверьте путь к FFHQ."
        )

    # Детерминированная сортировка для совпадения сплитов между процессами DDP
    paths.sort()
    return paths


class FFHQDataset(Dataset):
    """Датасет FFHQ 128x128 с поддержкой in-memory RAM кэширования."""

    def __init__(
        self,
        root: str | Path | None = None,
        paths: Sequence[str] | None = None,
        split: str = "train",
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        val_frac: float = 0.05,
        in_memory: bool = True,
        image_size: int = 128,
    ) -> None:
        self.transform = transform if transform is not None else self._default_transform()
        self.in_memory = in_memory
        self.image_size = (image_size, image_size)

        if paths is not None:
            all_paths = list(paths)
        elif root is not None:
            all_paths = scan_image_files(root)
        else:
            raise ValueError("Необходимо передать 'root' или готовый список 'paths'.")

        n_val = max(1, int(len(all_paths) * val_frac)) if val_frac > 0 else 0
        n_train = len(all_paths) - n_val

        if split == "train":
            self.paths = all_paths[:n_train]
        elif split == "val":
            self.paths = all_paths[n_train:]
        elif split == "all":
            self.paths = all_paths
        else:
            raise ValueError(f"Неизвестный split: {split!r}")

        self.cached_tensors: torch.Tensor | None = None
        if self.in_memory:
            self._preload_to_ram()

    def _preload_to_ram(self) -> None:
        """Предзагрузка всех изображений в один непрерывный uint8 тензор (RAM)."""
        total = len(self.paths)
        self.cached_tensors = torch.empty(
            (total, 3, self.image_size[0], self.image_size[1]),
            dtype=torch.uint8,
        )

        for i, path in enumerate(self.paths):
            try:
                img = io.read_image(path, mode=io.ImageReadMode.RGB)
                if img.shape[1:] != self.image_size:
                    img = transforms.functional.resize(
                        img, list(self.image_size), antialias=True
                    )
                self.cached_tensors[i] = img
            except Exception:
                with Image.open(path) as pil_img:
                    pil_img = pil_img.convert("RGB").resize(self.image_size)
                    self.cached_tensors[i] = io.image.pil_to_tensor(pil_img)

    @staticmethod
    def _default_transform() -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.in_memory and self.cached_tensors is not None:
            tensor = self.cached_tensors[idx]
        else:
            try:
                tensor = io.read_image(self.paths[idx], mode=io.ImageReadMode.RGB)
            except Exception:
                with Image.open(self.paths[idx]) as img:
                    tensor = io.image.pil_to_tensor(img.convert("RGB"))

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor


def get_dataloaders(
    data_root: str | Path,
    batch_size: int = 64,
    num_workers: int = 2,
    val_frac: float = 0.05,
    pin_memory: bool = True,
    distributed: bool = False,
    seed: int = 42,
    in_memory: bool = True,
    image_size: int = 128,
) -> tuple[
    DataLoader,
    DataLoader | None,
    DistributedSampler | None,
    DistributedSampler | None,
]:
    """Создаёт DataLoader'ы для train и val без повторного сканирования диска."""
    all_paths = scan_image_files(data_root)

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_ds = FFHQDataset(
        paths=all_paths,
        split="train",
        val_frac=val_frac,
        transform=train_transform,
        in_memory=in_memory,
        image_size=image_size,
    )

    val_ds = (
        FFHQDataset(
            paths=all_paths,
            split="val",
            val_frac=val_frac,
            transform=val_transform,
            in_memory=in_memory,
            image_size=image_size,
        )
        if val_frac > 0
        else None
    )

    train_sampler: DistributedSampler | None = None
    val_sampler: DistributedSampler | None = None

    if distributed:
        train_sampler = DistributedSampler(
            train_ds,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
        if val_ds is not None:
            val_sampler = DistributedSampler(
                val_ds,
                shuffle=False,
                seed=seed,
                drop_last=False,
            )

    effective_workers = 0 if in_memory else num_workers

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=effective_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=(effective_workers > 0),
        drop_last=True,
    )

    val_loader = (
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=effective_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            persistent_workers=(effective_workers > 0),
            drop_last=False,
        )
        if val_ds is not None
        else None
    )

    return train_loader, val_loader, train_sampler, val_sampler
