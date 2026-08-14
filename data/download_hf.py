"""
Загрузка FFHQ 70k через Hugging Face (streaming) с ресайзом до 128x128 на лету.

Датасет marcosv/ffhq-dataset содержит PNG в 1024x1024 (~90 GB суммарно),
поэтому используем streaming=True — скачиваем и сразу сохраняем в 128x128,
не держа полноразмерные файлы на диске.

Запуск из корня проекта:
    python data/download_hf.py
    python data/download_hf.py --n 5000   # только 5000 (тест)
    python data/download_hf.py --workers 8
"""

import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import io


def parse_args():
    p = argparse.ArgumentParser(description="Download FFHQ 128x128 from HuggingFace (streaming)")
    p.add_argument("--dataset",  type=str, default="marcosv/ffhq-dataset",
                   help="HuggingFace dataset repo id")
    p.add_argument("--split",    type=str, default="train")
    p.add_argument("--out_dir",  type=str, default="./data/thumbnails128x128")
    p.add_argument("--size",     type=int, default=128, help="Output image size (default: 128)")
    p.add_argument("--n",        type=int, default=None, help="Limit number of images")
    p.add_argument("--workers",  type=int, default=4,   help="Parallel save workers")
    return p.parse_args()


def save_image(args_tuple):
    """Сохраняет одно изображение: ресайз → PNG."""
    i, img, out_dir, size = args_tuple
    subdir = out_dir / f"{(i // 1000) * 1000:05d}"
    subdir.mkdir(parents=True, exist_ok=True)
    img_path = subdir / f"{i:05d}.png"
    if img_path.exists():
        return i, True  # уже скачано
    try:
        if not hasattr(img, "save"):
            # Если bytes — открываем через PIL
            img = Image.open(io.BytesIO(img))
        img = img.convert("RGB")
        if img.width != size or img.height != size:
            img = img.resize((size, size), Image.LANCZOS)
        img.save(img_path, format="PNG", optimize=True)
        return i, True
    except Exception as e:
        return i, f"ERROR: {e}"


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 Датасет: {args.dataset}")
    print(f"   Выход: {out_dir}  (размер: {args.size}×{args.size})")
    print(f"   Лимит: {'все ~70 000' if args.n is None else args.n} изображений")
    print(f"   Воркеры: {args.workers}\n")

    from datasets import load_dataset

    # Streaming — не скачиваем всё сразу
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    # Определяем колонку с изображением по первому элементу
    first = next(iter(ds))
    img_col = None
    for c in ["image", "img", "pixel_values", "jpg", "png"]:
        if c in first:
            img_col = c
            break
    if img_col is None:
        print(f"❌ Нет колонки с изображением. Доступные: {list(first.keys())}")
        sys.exit(1)
    print(f"  Колонка изображений: '{img_col}'\n")

    # Перебираем датасет батчами и сохраняем параллельно
    saved = 0
    errors = 0
    batch = []
    BATCH_SIZE = args.workers * 4

    for i, sample in enumerate(ds):
        if args.n and i >= args.n:
            break

        img = sample[img_col]
        batch.append((i, img, out_dir, args.size))

        if len(batch) >= BATCH_SIZE:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(save_image, item) for item in batch]
                for f in as_completed(futures):
                    idx, result = f.result()
                    if result is True:
                        saved += 1
                    else:
                        errors += 1
                        print(f"  ⚠️  img {idx}: {result}")
            batch = []

            if saved % 1000 < BATCH_SIZE:
                total_str = str(args.n) if args.n else "~70000"
                print(f"  [{saved}/{total_str}] ✓ сохранено  (ошибок: {errors})")

    # Остаток
    if batch:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(save_image, item) for item in batch]
            for f in as_completed(futures):
                idx, result = f.result()
                if result is True:
                    saved += 1
                else:
                    errors += 1

    print(f"\n✅ Готово! {saved} изображений → {out_dir}/")
    if errors:
        print(f"   ⚠️  Ошибок: {errors}")


if __name__ == "__main__":
    main()
