"""
Загрузка изображений для VAE, GAN и DDPM.

Поддерживает:

- Single-GPU.
- DistributedDataParallel.
- RGB.
- resize до image_size.
- train/validation split.
- deterministic split.
- RandomHorizontalFlip для train.
- нормализацию в [-1, 1].
- in-memory RAM cache.
- обычную загрузку с диска.
- старый ImageDataset API.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image

import torch
import torchvision.io as io
from PIL import Image
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
)
import torchvision.io as io
from torchvision import transforms


VALID_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
    }
)


# ============================================================================
# File discovery
# ============================================================================


def scan_image_files(
    root: Path | str,
) -> list[str]:
    """
    Рекурсивно сканирует директорию.

    Результат сортируется, чтобы на всех DDP rank
    порядок файлов был одинаковым.
    """

    root_path = (
        Path(root)
        .expanduser()
        .resolve()
    )

    if not root_path.is_dir():
        raise FileNotFoundError(
            "Директория с датасетом не найдена: "
            f"{root_path}"
        )

    paths: list[str] = []

    for dirpath, _, filenames in os.walk(
        root_path
    ):
        for filename in filenames:
            extension = (
                os.path.splitext(
                    filename
                )[1]
                .lower()
            )

            if extension in VALID_EXTENSIONS:
                paths.append(
                    os.path.join(
                        dirpath,
                        filename,
                    )
                )

    paths.sort()

    if not paths:
        raise FileNotFoundError(
            "Изображения не найдены в "
            f"{root_path}. "
            "Проверьте путь к датасету."
        )

    return paths


# ============================================================================
# Image loading
# ============================================================================


def _load_image(
    path: str,
    image_size: int,
) -> torch.Tensor:
    """
    Загружает изображение как uint8 tensor:

        C x H x W

    в RGB.
    """

    try:
        tensor = io.read_image(
            path,
            mode=io.ImageReadMode.RGB,
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
            image = image.convert(
                "RGB"
            )

            image = image.resize(
                (
                    image_size,
                    image_size,
                ),
                Image.Resampling.LANCZOS,
            )

            return torch.from_numpy(
                __import__(
                    "numpy"
                ).array(image)
            ).permute(
                2,
                0,
                1,
            ).contiguous()


# ============================================================================
# Transforms
# ============================================================================


def default_train_transform(
    image_size: int = 128,
) -> transforms.Compose:
    """
    Transform для train.

    Output:
        float32 tensor in [-1, 1].
    """

    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(
                p=0.5
            ),
            transforms.ConvertImageDtype(
                torch.float32
            ),
            transforms.Normalize(
                mean=[
                    0.5,
                    0.5,
                    0.5,
                ],
                std=[
                    0.5,
                    0.5,
                    0.5,
                ],
            ),
        ]
    )


def default_val_transform(
    image_size: int = 128,
) -> transforms.Compose:
    """
    Transform для validation.

    Без случайных аугментаций.
    """

    return transforms.Compose(
        [
            transforms.ConvertImageDtype(
                torch.float32
            ),
            transforms.Normalize(
                mean=[
                    0.5,
                    0.5,
                    0.5,
                ],
                std=[
                    0.5,
                    0.5,
                    0.5,
                ],
            ),
        ]
    )


# ============================================================================
# Dataset
# ============================================================================


class FFHQDataset(Dataset):
    """
    Основной dataset.

    Поддерживает:

        root=...
        paths=...
        split="train" / "val" / "all"

    При in_memory=True все изображения
    хранятся как uint8 в RAM.

    Важно:

    transforms выполняются после получения
    изображения из RAM.
    """


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
        transform: Callable[
            [torch.Tensor],
            torch.Tensor,
        ]
        | None = None,
        val_frac: float = 0.05,
        in_memory: bool = True,
        image_size: int = 128,
    ) -> None:
        super().__init__()

        if not 0.0 <= val_frac < 1.0:
            raise ValueError(
                "val_frac должен быть "
                "в диапазоне [0, 1)."
            )

        if image_size <= 0:
            raise ValueError(
                "image_size должен быть > 0."
            )

        self.image_size = (
            image_size,
            image_size,
        )

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

        if paths is not None:
            all_paths = list(paths)

        elif root is not None:
            all_paths = scan_image_files(
                root
            )

            self._indices = torch.tensor(
                [
                    all_path_to_index[path]
                    for path in self.paths
                ],
                dtype=torch.long,
            )
        else:
            raise ValueError(
                "Необходимо передать "
                "'root' или 'paths'."
            )

        if not all_paths:
            raise ValueError(
                "Dataset пуст."
            )

        all_paths.sort()

        n_total = len(all_paths)

        if val_frac > 0:
            n_val = max(
                1,
                int(
                    n_total * val_frac
                ),
            )
        else:
            n_val = 0

        n_train = n_total - n_val

        if n_train <= 0:
            raise ValueError(
                "После validation split "
                "train dataset пуст."
            )

        if split == "train":
            selected_paths = (
                all_paths[:n_train]
            )

        elif split == "val":
            selected_paths = (
                all_paths[n_train:]
            )

        elif split == "all":
            selected_paths = all_paths

        else:
            raise ValueError(
                f"Неизвестный split: {split!r}. "
                "Используйте train, val или all."
            )

        self.paths = selected_paths

        self.cached_tensors: (
            torch.Tensor | None
        ) = None

        if self.in_memory:
            self._preload_to_ram()

    # ---------------------------------------------------------------------

    def _preload_to_ram(self) -> None:
        """
        Загружает dataset в RAM.

        uint8 вместо float32 экономит RAM в 4 раза.

        Нормализация выполняется только после
        извлечения конкретного изображения.
        """

        total = len(self.paths)

        self.cached_tensors = torch.empty(
            (
                total,
                3,
                self.image_size[0],
                self.image_size[1],
            ),
            dtype=torch.uint8,
        )

        for index, path in enumerate(
            self.paths
        ):
            self.cached_tensors[
                index
            ] = _load_image(
                path,
                self.image_size[0],
            )

    # ---------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.paths)

    # ---------------------------------------------------------------------

    def __getitem__(
        self,
        index: int,
    ) -> torch.Tensor:
        if (
            self.in_memory
            and self.cached_tensors
            is not None
        ):
            tensor = self.cached_tensors[
                index
            ]

        else:
            tensor = _load_image(
                self.paths[index],
                self.image_size[0],
            )

        if self.transform is not None:
            tensor = self.transform(
                tensor
            )

        return tensor


# ============================================================================
# Backward-compatible ImageDataset
# ============================================================================


class ImageDataset(FFHQDataset):
    """
    Совместимый старый интерфейс.

    Например:

        dataset = ImageDataset(
            "./data",
            image_size=128,
        )

    По умолчанию возвращает train-style
    изображения в [-1, 1].
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 128,
        transform: Callable[
            [torch.Tensor],
            torch.Tensor,
        ]
        | None = None,
        in_memory: bool = True,
        **kwargs,
    ):
        if transform is None:
            transform = default_train_transform(
                image_size
            )

        super().__init__(
            root=root,
            split="all",
            transform=transform,
            val_frac=0.0,
            in_memory=in_memory,
            image_size=image_size,
            **kwargs,
        )


# ============================================================================
# Dataloader factory
# ============================================================================


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
    """
    Создаёт train/validation DataLoader.

    Важные свойства для DDP:

    - scan_image_files выполняется один раз
      на каждом rank;
    - порядок paths одинаковый благодаря sort();
    - DistributedSampler получает одинаковый
      dataset;
    - sampler.set_epoch(epoch) вызывается
      в train.py;
    - каждый rank получает собственную
      часть dataset.

    При in_memory=True workers отключаются,
    чтобы не создавать дополнительные копии
    большого RAM cache через multiprocessing.
    """

    all_paths = scan_image_files(
        data_root
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
        )

    train_sampler: (
        DistributedSampler | None
    ) = None

    val_sampler: (
        DistributedSampler | None
    ) = None

    if distributed:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=True,
            seed=seed,
            drop_last=True,
        )

        if val_ds is not None:
            val_sampler = DistributedSampler(
                val_ds,
                num_replicas=(
                    torch.distributed.get_world_size()
                ),
                rank=(
                    torch.distributed.get_rank()
                ),
                shuffle=False,
                seed=seed,
                drop_last=False,
            )

    if in_memory:
        effective_workers = 0

    else:
        effective_workers = max(
            0,
            num_workers,
        )
    )

    use_pin_memory = (
        pin_memory
        and torch.cuda.is_available()
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


# ============================================================================
# Smoke test
# ============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        required=True,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    train_loader, val_loader, _, _ = (
        get_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=0,
            in_memory=True,
            val_frac=0.05,
        )
    )

    images = next(
        iter(train_loader)
    )

    print(
        "Train batch:",
        tuple(images.shape),
    )

    print(
        "dtype:",
        images.dtype,
    )

    print(
        "min:",
        images.min().item(),
    )

    print(
        "max:",
        images.max().item(),
    )

    print(
        "Train samples:",
        len(
            train_loader.dataset
        ),
    )

    if val_loader is not None:
        print(
            "Val samples:",
            len(
                val_loader.dataset
            ),
        )
