# Image Generation: VAE vs GAN vs DDPM

Сравнение трёх архитектур генерации изображений на датасете **FFHQ 128×128**.

| Модель | Качество | Скорость обучения | Стабильность |
|--------|----------|-------------------|--------------|
| VAE    | Размытые картинки | ⚡ Быстро | ✅ Стабильно |
| GAN    | Резкие картинки | ⚡ Быстро | ⚠️ Mode collapse |
| DDPM   | Лучшее качество | 🐢 Медленно | ✅ Стабильно |

## Структура проекта

```
ImageModel/
├── VAE/
│   ├── vae.py          # Архитектура β-VAE
│   └── train.py        # Скрипт обучения
├── GAN/
│   ├── gan.py          # Архитектура DCGAN (SpectralNorm + Self-Attention)
│   └── train.py        # Скрипт обучения
├── DDPM/
│   ├── ddpm.py         # UNet + DDPM (Cosine Schedule)
│   └── train.py        # Скрипт обучения
├── infer/
│   └── infer.py        # Единый скрипт инференса для всех моделей
├── configs/
│   ├── vae.yaml        # Конфигурация VAE
│   ├── gan.yaml        # Конфигурация GAN
│   └── ddpm.yaml       # Конфигурация DDPM
├── data/               # Датасет FFHQ (thumbnails128x128/)
└── checkpoints/        # Сохранённые чекпоинты (создаётся при обучении)
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirments.txt
```

## Данные

### 1. Скачивание изображений FFHQ 128x128 (Kaggle)
Для скачивания всех 70 000 изображений (размер архива ~2 ГБ) запустите:
```bash
python data/download_kaggle.py
```
Изображения сохраняются в `data/thumbnails128x128/` (с разбиением по подпапкам `00000/`, `01000/`, ...).

### 2. Скачивание текстовых описаний (Captions)
Для скачивания текстовых LLaVA-описаний для картинок:
```bash
python data/save_captions.py
```
Описания сохраняются в `data/captions.json` и строго соответствуют ID файлов (`00000`, `00001`, ...).

---

## Обучение

Все скрипты запускаются **из корня проекта**.

### VAE (β-VAE)

```bash
# Через конфиг (рекомендуется)
python VAE/train.py --config configs/vae.yaml

# Вручную задать параметры
python VAE/train.py --data_root ./data --epochs 50 --batch_size 32

# Продолжить с чекпоинта
python VAE/train.py --resume checkpoints/vae/checkpoint_ep020.pt

# Быстрая проверка кода (без данных, 2 мини-эпохи)
python VAE/train.py --dry_run
```

Ключевые параметры (`configs/vae.yaml`):

| Параметр | Значение по умолчанию | Описание |
|----------|-----------------------|----------|
| `latent_dim` | 256 | Размерность латентного пространства |
| `beta` | 4.0 | β-VAE: 1 = vanilla VAE, >1 = disentangled |
| `epochs` | 50 | Число эпох |
| `batch_size` | 32 | Размер батча |
| `lr` | 1e-4 | Learning rate |

---

### GAN (DCGAN + SpectralNorm + Self-Attention)

```bash
# Через конфиг (рекомендуется)
python GAN/train.py --config configs/gan.yaml

# Вручную задать параметры
python GAN/train.py --data_root ./data --epochs 100 --batch_size 32

# Продолжить с чекпоинта
python GAN/train.py --resume checkpoints/gan/checkpoint_ep050.pt

# Быстрая проверка кода
python GAN/train.py --dry_run
```

Ключевые параметры (`configs/gan.yaml`):

| Параметр | Значение по умолчанию | Описание |
|----------|-----------------------|----------|
| `latent_dim` | 100 | Размерность шума z |
| `ngf` / `ndf` | 64 | Базовое число фильтров G / D |
| `epochs` | 100 | Число эпох |
| `batch_size` | 32 | Размер батча |
| `lr_g` / `lr_d` | 2e-4 | Learning rate Generator / Discriminator |
| `label_smooth` | 0.1 | Label smoothing (реальные → [0.9, 1.0]) |

---

### DDPM (UNet + Self-Attention + Cosine Schedule)

```bash
# Через конфиг (рекомендуется)
python DDPM/train.py --config configs/ddpm.yaml

# Вручную задать параметры
python DDPM/train.py --data_root ./data --epochs 200 --batch_size 16

# Продолжить с чекпоинта
python DDPM/train.py --resume checkpoints/ddpm/checkpoint_ep100.pt

# Быстрая проверка кода
python DDPM/train.py --dry_run
```

Ключевые параметры (`configs/ddpm.yaml`):

| Параметр | Значение по умолчанию | Описание |
|----------|-----------------------|----------|
| `base_channels` | 64 | Базовое число каналов UNet |
| `timesteps` | 1000 | T — количество шагов диффузии |
| `schedule` | cosine | Расписание шума: `cosine` или `linear` |
| `ema_decay` | 0.9999 | EMA весов модели (улучшает качество) |
| `epochs` | 200 | Число эпох |
| `batch_size` | 16 | Меньше из-за большей памяти UNet |

> **Замечание:** Один шаг семплирования DDPM требует T=1000 forward-пассов через UNet.
> Параметр `sample_every` лучше ставить 20+.

---

## Инференс

Единый скрипт `infer/infer.py` поддерживает все три модели.

### Генерация из одной модели

```bash
# VAE
python infer/infer.py --model vae --checkpoint checkpoints/vae/best.pt --n_samples 64

# GAN
python infer/infer.py --model gan --checkpoint checkpoints/gan/best.pt --n_samples 64

# DDPM (медленно — T=1000 шагов)
python infer/infer.py --model ddpm --checkpoint checkpoints/ddpm/best.pt --n_samples 16
```

Результат сохраняется в `infer/<model>_samples.png` по умолчанию.

Указать путь сохранения вручную:
```bash
python infer/infer.py --model vae --checkpoint checkpoints/vae/best.pt \
    --n_samples 64 --nrow 8 --output my_results/vae_gen.png
```

### Сравнительный постер всех трёх моделей

```bash
python infer/infer.py --compare \
    --vae_ckpt  checkpoints/vae/best.pt \
    --gan_ckpt  checkpoints/gan/best.pt \
    --ddpm_ckpt checkpoints/ddpm/best.pt
```

Результат сохраняется в `checkpoints/comparison.png`.

### Параметры инференса

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--model` | — | Модель: `vae`, `gan` или `ddpm` |
| `--checkpoint` | `checkpoints/<model>/best.pt` | Путь к чекпоинту |
| `--n_samples` | 64 | Количество генерируемых изображений |
| `--nrow` | 8 | Изображений в строке сетки |
| `--output` | `infer/<model>_samples.png` | Путь для сохранения |
| `--device` | auto | `cuda`, `cpu` или `auto` |
| `--seed` | 42 | Random seed |
| `--compare` | — | Режим сравнения всех трёх моделей |

---

## Результаты и артефакты

После обучения в `checkpoints/<model>/` появятся:

```
checkpoints/
├── vae/
│   ├── best.pt                  # Лучший чекпоинт
│   ├── checkpoint_ep<N>.pt      # Периодические чекпоинты
│   ├── samples/gen_ep<N>.png    # Примеры генерации по эпохам
│   └── metrics/                 # CSV-логи и графики потерь
├── gan/
│   └── ...
└── ddpm/
    └── ...
```

---

## Математика

### VAE — ELBO

$$ELBO = \mathbb{E}_{q}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

### GAN — Minimax

$$\min_G \max_D \; \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))]$$

### DDPM — MSE на шум

$$L = \mathbb{E}_{t,\, x_0,\, \epsilon}\bigl[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\bigr]$$
