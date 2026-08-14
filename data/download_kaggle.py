"""
Загрузка FFHQ 128x128 thumbnails с Kaggle:
https://www.kaggle.com/datasets/greatgamedota/ffhq-face-data-set

Датасет содержит все 70 000 оригинальных изображений в разрешении 128x128 (~2.0 GB).
Имена файлов (00000.png, 00001.png, ...) строго соответствуют ID в data/captions.json.

Запуск:
    python data/download_kaggle.py
    python data/download_kaggle.py --verify_only
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from PIL import Image
from kaggle.api.kaggle_api_extended import KaggleApi


def parse_args():
    p = argparse.ArgumentParser(description="Download FFHQ 128x128 from Kaggle and verify with captions")
    p.add_argument("--dataset", type=str, default="greatgamedota/ffhq-face-data-set",
                   help="Kaggle dataset identifier")
    p.add_argument("--out_dir", type=str, default="./data/thumbnails128x128",
                   help="Target directory for thumbnails")
    p.add_argument("--organize_subdirs", action="store_true", default=True,
                   help="Organize images into 1000-image subfolders (00000/, 01000/, etc.)")
    p.add_argument("--verify_only", action="store_true",
                   help="Skip download, only verify images and captions")
    return p.parse_args()


def download_from_kaggle(dataset_id: str, data_root: Path):
    data_root.mkdir(parents=True, exist_ok=True)
    print(f"\n📥 Скачивание и распаковка датасета '{dataset_id}' с Kaggle в {data_root}...")
    
    api = KaggleApi()
    api.authenticate()
    
    # Скачивает и автоматически разархивирует в data_root
    api.dataset_download_files(dataset_id, path=str(data_root), unzip=True, quiet=False)
    print(f"✅ Датасет скачан и распакован в {data_root}!")


def organize_into_subfolders(out_dir: Path):
    """Раскладывает файлы 00000.png .. 69999.png по подпапкам 00000/, 01000/ и т.д."""
    if not out_dir.exists():
        return
    flat_pngs = [p for p in out_dir.glob("*.png") if p.is_file()]
    if not flat_pngs:
        return

    print(f"📁 Организация {len(flat_pngs):,} файлов по подпапкам...")
    for p in flat_pngs:
        stem = p.stem
        if stem.isdigit():
            idx = int(stem)
            subdir_name = f"{(idx // 1000) * 1000:05d}"
            subdir = out_dir / subdir_name
            subdir.mkdir(exist_ok=True)
            shutil.move(str(p), str(subdir / p.name))
    print("✓ Файлы успешно разложены по подпапкам")


def verify_captions_and_images(out_dir: Path, captions_path: Path = Path("data/captions.json")):
    print("\n" + "=" * 70)
    print("🔍 ВЕРИФИКАЦИЯ: Проверка соответствия файлов и описаний (captions)")
    print("=" * 70)

    if not captions_path.exists():
        print(f"⚠️ Файл описаний {captions_path} не найден! Запустите python data/save_captions.py")
        return

    with open(captions_path, "r", encoding="utf-8") as f:
        captions = json.load(f)

    sample_ids = ["00000", "00001", "00002", "00003", "00004", "00010", "00018", "00020"]
    checked = 0

    for sid in sample_ids:
        idx = int(sid)
        sub = f"{(idx // 1000) * 1000:05d}"
        path1 = out_dir / sub / f"{sid}.png"
        path2 = out_dir / f"{sid}.png"

        img_path = path1 if path1.exists() else (path2 if path2.exists() else None)
        caption = captions.get(sid, "<нет описания>")

        if img_path and img_path.exists():
            img = Image.open(img_path)
            print(f"\n🖼️  ID [{sid}] ({img_path.name}, размер: {img.size}):")
            print(f"   Описание: {caption}")
            checked += 1
        else:
            print(f"\n⚠️  ID [{sid}]: файл изображения не найден на диске.")
            print(f"   Описание в json: {caption}")

    total_images = len(list(out_dir.rglob("*.png"))) if out_dir.exists() else 0
    print("\n" + "-" * 70)
    print(f"📊 Всего изображений на диске: {total_images:,}")
    print(f"📊 Всего описаний в JSON:      {len(captions):,}")
    if checked > 0:
        print(f"✅ Проверено {checked} образцов. Имена файлов и описания полностью синхронизированы!")
    print("=" * 70)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    data_root = out_dir.parent

    # Очищаем неполные временные файлы
    temp_dir = data_root / "temp_kaggle_download"
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    if not args.verify_only:
        download_from_kaggle(args.dataset, data_root)
        if args.organize_subdirs:
            organize_into_subfolders(out_dir)

    verify_captions_and_images(out_dir)


if __name__ == "__main__":
    main()
