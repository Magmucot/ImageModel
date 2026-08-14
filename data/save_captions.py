"""
Сохраняет FFHQ LLaVA-captions в data/captions.json.
Использует PyArrow и fsspec для скачивания только текстовой колонки без тяжелых изображений.
Формат: { "00000": "The image features a blonde woman...", "00001": "...", ... }

Запуск:
    python data/save_captions.py
"""

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fsspec
import pyarrow.parquet as pq
from tqdm import tqdm


def get_parquet_files(repo_id: str = "irodkin/ffhq_with_llava_shorter_captions") -> list[str]:
    api_url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main/data"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        items = json.loads(resp.read().decode("utf-8"))
    parquet_files = sorted([item["path"] for item in items if item["path"].endswith(".parquet")])
    return parquet_files


def download_shard_captions(shard_idx: int, shard_path: str, repo_id: str, max_retries: int = 3):
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{shard_path}"
    for attempt in range(max_retries):
        try:
            with fsspec.open(url, mode="rb") as f:
                table = pq.read_table(f, columns=["text"])
                texts = table["text"].to_pylist()
                return shard_idx, texts
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Ошибка загрузки шарда {shard_path} после {max_retries} попыток: {e}")
            time.sleep(2 * (attempt + 1))


def main():
    repo_id = "irodkin/ffhq_with_llava_shorter_captions"
    out_file = Path("data/captions.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"📥 Получаем список файлов из {repo_id}...")
    parquet_files = get_parquet_files(repo_id)
    print(f"✓ Найдено {len(parquet_files)} parquet-файлов (шардов)")

    print("🚀 Скачиваем текстовые описания (без картинок)...")
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(download_shard_captions, i, path, repo_id)
            for i, path in enumerate(parquet_files)
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Загрузка шардов"):
            results.append(f.result())

    # Сортируем по исходному порядку шардов
    results.sort(key=lambda x: x[0])

    captions = {}
    idx = 0
    for _, shard_texts in results:
        for text in shard_texts:
            img_id = f"{idx:05d}"
            captions[img_id] = text
            idx += 1

    print(f"\n💾 Сохраняем {len(captions):,} описаний в {out_file}...")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)

    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"✅ Готово! Файл сохранен: {out_file} ({file_size_mb:.2f} MB)")
    print("\nПримеры:")
    for sample_id in ["00000", "00001", "00002"]:
        if sample_id in captions:
            print(f"  [{sample_id}]: {captions[sample_id]}")


if __name__ == "__main__":
    main()
