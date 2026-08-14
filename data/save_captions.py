"""
Сохраняет FFHQ LLaVA-captions в data/captions.json
Формат: { "00000": "A young woman with...", "00001": "...", ... }

Запуск:
    python data/save_captions.py
"""
import json
from pathlib import Path
from datasets import load_dataset

print("📥 Загружаю FFHQ captions (LLaVA)...")
ds = load_dataset("irodkin/ffhq_with_llava_shorter_captions", split="train")
print(f"✓ {len(ds)} записей | колонки: {ds.column_names}")

# Смотрим первый элемент чтобы понять структуру
sample = ds[0]
print(f"\nПример записи: {sample}\n")

# Определяем колонку с ID и с текстом
# Возможные имена: 'caption', 'text', 'llava_caption', 'shorter_caption'
text_col = None
id_col   = None
for c in ds.column_names:
    if c in ("caption", "text", "llava_caption", "shorter_caption", "description"):
        text_col = c
    if c in ("id", "idx", "image_id", "file_name", "filename"):
        id_col = c

print(f"id_col={id_col!r}, text_col={text_col!r}")

captions = {}
for i, row in enumerate(ds):
    img_id = str(row[id_col]).zfill(5) if id_col else f"{i:05d}"
    caption = row[text_col] if text_col else str(row)
    captions[img_id] = caption

out = Path("data/captions.json")
out.write_text(json.dumps(captions, ensure_ascii=False, indent=None))
print(f"\n✅ Сохранено {len(captions)} captions → {out}")
print(f"   Пример [00000]: {list(captions.values())[0][:120]}...")
