"""
Загрузка данных FFHQ (Flickr-Faces-HQ) для обучения VAE, GAN, DDPM.
Поддерживает 128x128 thumbnails.
"""

import os
import glob
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Датасет
# ─────────────────────────────────────────────────────────────────────────────

class FFHQDataset(Dataset):
    """
    Датасет FFHQ для изображений 128x128.
    Ожидает структуру:
        data/
          thumbnails128x128/
            00000/  *.png
            01000/  *.png
            ...
    """

    EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

    def __init__(self, root: str, split: str = "train", transform=None, val_frac: float = 0.05):
        """
        Args:
            root:       путь к папке data/
            split:      'train' | 'val' | 'all'
            transform:  torchvision-трансформации; если None — применяются дефолтные
            val_frac:   доля валидационных данных
        """
        self.root = Path(root)
        self.transform = transform or self._default_transform()

        # Собираем все пути к картинкам
        all_paths = self._gather_images()
        if len(all_paths) == 0:
            raise FileNotFoundError(
                f"Изображения не найдены в {self.root}. "
                "Убедитесь, что FFHQ thumbnails128x128 скачаны."
            )

        # Разделяем на train/val детерминированно
        n_val = max(1, int(len(all_paths) * val_frac))
        n_train = len(all_paths) - n_val

        if split == "train":
            self.paths = all_paths[:n_train]
        elif split == "val":
            self.paths = all_paths[n_train:]
        else:  # 'all'
            self.paths = all_paths

        print(f"[Dataset] split={split!r}, images={len(self.paths):,}")

    def _gather_images(self):
        """Рекурсивно ищет все изображения в root."""
        paths = []
        for ext in self.EXTENSIONS:
            # thumbnails128x128/**/*.png
            paths += sorted(self.root.rglob(f"*{ext}"))
            paths += sorted(self.root.rglob(f"*{ext.upper()}"))
        # Убираем дубли и сортируем
        paths = sorted(set(paths))
        return [str(p) for p in paths]

    @staticmethod
    def _default_transform():
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# ─────────────────────────────────────────────────────────────────────────────
# Фабрика DataLoader-ов
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    val_frac: float = 0.05,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Возвращает (train_loader, val_loader) для FFHQ.

    Args:
        data_root:   путь к папке data/ (содержит thumbnails128x128/)
        batch_size:  размер батча
        num_workers: количество воркеров DataLoader
        val_frac:    доля валидации
        pin_memory:  ускорение для GPU

    Returns:
        train_loader, val_loader
    """
    train_ds = FFHQDataset(data_root, split="train", val_frac=val_frac)
    val_ds   = FFHQDataset(data_root, split="val",   val_frac=val_frac)

    # Для val — без аугментации
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    val_ds.transform = val_transform

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Тест
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./", help="путь к папке data/")
    args = parser.parse_args()

    train_loader, val_loader = get_dataloaders(args.data_root, batch_size=8, num_workers=2)
    batch = next(iter(train_loader))
    print(f"Батч: {batch.shape}, min={batch.min():.3f}, max={batch.max():.3f}")
    print(f"Train батчей: {len(train_loader)}, Val батчей: {len(val_loader)}")
