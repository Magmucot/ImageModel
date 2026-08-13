"""
Единый скрипт инференса для VAE, GAN и DDPM.

Использование:
    # Генерация из VAE
    python infer/infer.py --model vae --checkpoint checkpoints/vae/best.pt --n_samples 64

    # Генерация из GAN
    python infer/infer.py --model gan --checkpoint checkpoints/gan/best.pt --n_samples 64

    # Генерация из DDPM (медленно — T=1000 шагов)
    python infer/infer.py --model ddpm --checkpoint checkpoints/ddpm/best.pt --n_samples 16

    # Сравнительный постер всех трёх моделей
    python infer/infer.py --compare \\
        --vae_ckpt checkpoints/vae/best.pt \\
        --gan_ckpt checkpoints/gan/best.pt \\
        --ddpm_ckpt checkpoints/ddpm/best.pt
"""

import sys
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from utils.utils import (
    save_sample_grid,
    load_checkpoint,
    compare_models,
    denormalize,
)


# ─────────────────────────────────────────────────────────────────────────────
# Аргументы
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Инференс для VAE / GAN / DDPM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Режим работы
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model",   type=str, choices=["vae", "gan", "ddpm"],
                      help="выбор одной модели для генерации")
    mode.add_argument("--compare", action="store_true",
                      help="сравнительный постер всех трёх моделей")

    # Чекпоинты
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="путь к чекпоинту (для --model)")
    parser.add_argument("--vae_ckpt",  type=str, default="checkpoints/vae/best.pt")
    parser.add_argument("--gan_ckpt",  type=str, default="checkpoints/gan/best.pt")
    parser.add_argument("--ddpm_ckpt", type=str, default="checkpoints/ddpm/best.pt")

    # Параметры генерации
    parser.add_argument("--n_samples",  type=int, default=64,
                        help="количество генерируемых изображений")
    parser.add_argument("--nrow",       type=int, default=8,
                        help="изображений в строке сетки")
    parser.add_argument("--output",     type=str, default=None,
                        help="путь для сохранения результата")

    # Параметры моделей (если нет в чекпоинте)
    parser.add_argument("--latent_dim",    type=int,   default=256)
    parser.add_argument("--beta",          type=float, default=4.0)
    parser.add_argument("--ngf",           type=int,   default=64)
    parser.add_argument("--gan_latent",    type=int,   default=100,
                        help="latent_dim для GAN")
    parser.add_argument("--base_channels", type=int,   default=64)
    parser.add_argument("--time_emb_dim",  type=int,   default=256)
    parser.add_argument("--timesteps",     type=int,   default=1000)
    parser.add_argument("--schedule",      type=str,   default="cosine")

    parser.add_argument("--device",  type=str, default="auto")
    parser.add_argument("--seed",    type=int, default=42)

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Загрузчики моделей
# ─────────────────────────────────────────────────────────────────────────────

def load_vae(checkpoint_path: str, args, device: torch.device) -> "VAE":
    from VAE.vae import VAE

    # Пробуем получить параметры из чекпоинта
    ckpt = load_checkpoint(checkpoint_path, device=str(device))
    ckpt_args = ckpt.get("args", {})
    latent_dim = ckpt_args.get("latent_dim", args.latent_dim)
    beta       = ckpt_args.get("beta",       args.beta)

    model = VAE(latent_dim=latent_dim, beta=beta).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  ✓ VAE загружен (latent_dim={latent_dim}, β={beta})")
    return model


def load_gan(checkpoint_path: str, args, device: torch.device) -> "Generator":
    from GAN.gan import Generator

    ckpt = load_checkpoint(checkpoint_path, device=str(device))
    ckpt_args  = ckpt.get("args", {})
    latent_dim = ckpt_args.get("latent_dim", args.gan_latent)
    ngf        = ckpt_args.get("ngf",        args.ngf)

    G = Generator(latent_dim=latent_dim, ngf=ngf).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()
    print(f"  ✓ Generator загружен (latent_dim={latent_dim}, ngf={ngf})")
    return G


def load_ddpm(checkpoint_path: str, args, device: torch.device):
    from DDPM.ddpm import UNet, DDPM as DDPMClass

    ckpt = load_checkpoint(checkpoint_path, device=str(device))
    ckpt_args = ckpt.get("args", {})
    base_ch   = ckpt_args.get("base_channels", args.base_channels)
    emb_dim   = ckpt_args.get("time_emb_dim",  args.time_emb_dim)
    timesteps = ckpt_args.get("timesteps",      args.timesteps)
    schedule  = ckpt_args.get("schedule",       args.schedule)

    unet = UNet(img_channels=3, base_channels=base_ch, time_emb_dim=emb_dim).to(device)

    # Предпочитаем EMA-веса если есть
    if "ema" in ckpt:
        unet.load_state_dict(ckpt["ema"])
        print(f"  ✓ DDPM загружен с EMA-весами (base_ch={base_ch}, T={timesteps})")
    else:
        unet.load_state_dict(ckpt["model"])
        print(f"  ✓ DDPM загружен (base_ch={base_ch}, T={timesteps})")

    ddpm = DDPMClass(unet, timesteps=timesteps, schedule=schedule, device=str(device))
    return ddpm


# ─────────────────────────────────────────────────────────────────────────────
# Генерация одной модели
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_vae(model, n: int, device: torch.device) -> torch.Tensor:
    return model.sample(n=n, device=str(device))


@torch.no_grad()
def generate_gan(G, n: int, device: torch.device, latent_dim: int = 100) -> torch.Tensor:
    z = torch.randn(n, latent_dim, 1, 1, device=device)
    return G(z)


@torch.no_grad()
def generate_ddpm(ddpm, n: int) -> torch.Tensor:
    return ddpm.sample(n=n, img_shape=(3, 128, 128), verbose=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n🔍 Inference | Device: {device}")

    # ── Режим сравнения всех трёх ─────────────────────────────────────────────
    if args.compare:
        print("\n📊 Режим сравнения VAE / GAN / DDPM")
        results = {}

        # VAE
        if os.path.exists(args.vae_ckpt):
            print("\n[1/3] VAE...")
            vae = load_vae(args.vae_ckpt, args, device)
            results["VAE"] = {
                "samples":      generate_vae(vae, 16, device),
                "loss_history": [],   # можно загрузить из CSV если нужно
            }
        else:
            print(f"  ⚠️  VAE чекпоинт не найден: {args.vae_ckpt}")

        # GAN
        if os.path.exists(args.gan_ckpt):
            print("\n[2/3] GAN...")
            ckpt_g = load_checkpoint(args.gan_ckpt, device=str(device))
            gan_ld = ckpt_g.get("args", {}).get("latent_dim", args.gan_latent)
            G = load_gan(args.gan_ckpt, args, device)
            results["GAN"] = {
                "samples":      generate_gan(G, 16, device, latent_dim=gan_ld),
                "loss_history": [],
            }
        else:
            print(f"  ⚠️  GAN чекпоинт не найден: {args.gan_ckpt}")

        # DDPM
        if os.path.exists(args.ddpm_ckpt):
            print("\n[3/3] DDPM (медленно)...")
            ddpm = load_ddpm(args.ddpm_ckpt, args, device)
            results["DDPM"] = {
                "samples":      generate_ddpm(ddpm, 16),
                "loss_history": [],
            }
        else:
            print(f"  ⚠️  DDPM чекпоинт не найден: {args.ddpm_ckpt}")

        if not results:
            print("❌ Ни один чекпоинт не найден. Завершение.")
            return

        output_path = args.output or "checkpoints/comparison.png"
        compare_models(results, save_path=output_path)
        print(f"\n✅ Сравнительный постер → {output_path}")
        return

    # ── Режим одной модели ────────────────────────────────────────────────────
    ckpt_path = args.checkpoint
    if not ckpt_path:
        # Дефолтный путь
        ckpt_path = f"checkpoints/{args.model}/best.pt"
    if not os.path.exists(ckpt_path):
        print(f"❌ Чекпоинт не найден: {ckpt_path}")
        return

    print(f"\n[{args.model.upper()}] Чекпоинт: {ckpt_path}")
    print(f"Генерируем {args.n_samples} изображений...")

    if args.model == "vae":
        model = load_vae(ckpt_path, args, device)
        samples = generate_vae(model, args.n_samples, device)

    elif args.model == "gan":
        ckpt_g = load_checkpoint(ckpt_path, device=str(device))
        gan_ld = ckpt_g.get("args", {}).get("latent_dim", args.gan_latent)
        G = load_gan(ckpt_path, args, device)
        samples = generate_gan(G, args.n_samples, device, latent_dim=gan_ld)

    elif args.model == "ddpm":
        print(f"⏳ DDPM семплинг занимает ~{args.timesteps} forward-пассов через UNet...")
        ddpm = load_ddpm(ckpt_path, args, device)
        samples = generate_ddpm(ddpm, args.n_samples)

    # Сохраняем результат
    output_path = args.output or f"infer/{args.model}_samples.png"
    save_sample_grid(
        samples,
        path=output_path,
        nrow=args.nrow,
        title=f"{args.model.upper()} Generated Samples (n={args.n_samples})",
    )
    print(f"\n✅ Результат сохранён → {output_path}")


if __name__ == "__main__":
    main()
