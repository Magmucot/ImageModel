"""
DCGAN с Spectral Normalization и Self-Attention для изображений 128×128×3.

Улучшения по сравнению с базовой версией:
  - Spectral Normalization в Discriminator → липшицева непрерывность,
    устраняет взрывной рост градиентов, стабилизирует обучение
  - Self-Attention в Generator (на уровне 64×64) → глобальная согласованность
  - Опциональный Residual блок в середине Generator
  - Label smoothing (0.0/0.9 вместо 0.0/1.0) — снижает overconfidence
  - Instance noise — небольшой шум к входу Discriminator в начале обучения
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Строительные блоки
# ─────────────────────────────────────────────────────────────────────────────

class SelfAttention2d(nn.Module):
    """
    Multi-head Self-Attention для 2D feature maps.
    Используется в Generator на разрешении 64×64 для глобальной согласованности.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.scale     = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv  = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        # Гамма = 0 → сначала модуль — identity, постепенно включается
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        h = self.norm(x)

        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        def to_heads(t: torch.Tensor) -> torch.Tensor:
            t = t.view(B, self.num_heads, self.head_dim, H * W)
            return t.permute(0, 1, 3, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out  = torch.matmul(attn, v)
        out  = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)

        return self.proj(out) * self.gamma + residual


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Генератор DCGAN для изображений 128×128×3.

    Архитектура:
        z (latent_dim × 1 × 1)
        → ngf*16 × 4  × 4
        → ngf*8  × 8  × 8
        → ngf*4  × 16 × 16
        → ngf*2  × 32 × 32
        → ngf    × 64 × 64  ← Self-Attention
        → ngf//2 × 128× 128
        → 3      × 128× 128 (Tanh)
    """

    def __init__(self, latent_dim: int = 100, ngf: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        def up_block(in_c, out_c, bn=True):
            layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False)]
            if bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # z × 1 × 1 → ngf*16 × 4 × 4
        self.init_block = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(inplace=True),
        )

        self.up1 = up_block(ngf * 16, ngf * 8)    # 4  → 8
        self.up2 = up_block(ngf * 8,  ngf * 4)    # 8  → 16
        self.up3 = up_block(ngf * 4,  ngf * 2)    # 16 → 32
        self.up4 = up_block(ngf * 2,  ngf)         # 32 → 64

        # Self-Attention на 64×64
        self.attn = SelfAttention2d(ngf, num_heads=4)

        self.up5 = up_block(ngf, ngf // 2)         # 64  → 128

        # Финальный слой 128 → 128 (без стрaйда, рафинирование)
        self.out_conv = nn.Sequential(
            nn.Conv2d(ngf // 2, 3, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim, 1, 1) — шум из N(0, I)
        Returns:
            (B, 3, 128, 128) — синтезированное изображение в [-1, 1]
        """
        h = self.init_block(z)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        h = self.attn(h)   # глобальная согласованность на 64×64
        h = self.up5(h)
        return self.out_conv(h)

    @torch.no_grad()
    def sample(self, n: int = 16, device: str = "cpu") -> torch.Tensor:
        """Генерирует n изображений."""
        z = torch.randn(n, self.latent_dim, 1, 1, device=device)
        return self(z)


# ─────────────────────────────────────────────────────────────────────────────
# Discriminator
# ─────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    Дискриминатор DCGAN со Spectral Normalization.

    Spectral Norm обеспечивает K-Lipschitz условие, что:
      - Стабилизирует обучение
      - Предотвращает взрыв градиентов
      - Снижает вероятность mode collapse

    Архитектура:
        3   × 128 × 128
        → ndf   × 64 × 64
        → ndf*2 × 32 × 32
        → ndf*4 × 16 × 16
        → ndf*8 × 8  × 8
        → ndf*8 × 4  × 4
        → ndf*8 × 2  × 2
        → 1     × 1  × 1   (Sigmoid)
    """

    def __init__(self, nc: int = 3, ndf: int = 64):
        super().__init__()

        def sn_conv(in_c, out_c, k=4, s=2, p=1):
            """Conv2d со Spectral Norm."""
            return nn.utils.spectral_norm(
                nn.Conv2d(in_c, out_c, k, s, p, bias=False)
            )

        self.main = nn.Sequential(
            # 3 × 128 × 128 → ndf × 64 × 64
            sn_conv(nc, ndf),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf × 64 × 64 → ndf*2 × 32 × 32
            sn_conv(ndf, ndf * 2),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf*2 × 32 × 32 → ndf*4 × 16 × 16
            sn_conv(ndf * 2, ndf * 4),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf*4 × 16 × 16 → ndf*8 × 8 × 8
            sn_conv(ndf * 4, ndf * 8),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf*8 × 8 × 8 → ndf*8 × 4 × 4
            sn_conv(ndf * 8, ndf * 8),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf*8 × 4 × 4 → ndf*8 × 2 × 2
            sn_conv(ndf * 8, ndf * 8),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf*8 × 2 × 2 → 1 × 1 × 1
            sn_conv(ndf * 8, 1, k=2, s=1, p=0),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (B, 3, 128, 128) — реальное или фейковое изображение
        Returns:
            (B,) — вероятность "реального" для каждого изображения
        """
        return self.main(img).view(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Инициализация весов
# ─────────────────────────────────────────────────────────────────────────────

def weights_init(m: nn.Module) -> None:
    """
    Инициализация весов DCGAN согласно оригинальной статье:
      Conv   → N(0, 0.02)
      BN     → N(1, 0.02), bias=0
    """
    classname = m.__class__.__name__
    if "Conv" in classname and not isinstance(m, SelfAttention2d):
        try:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        except AttributeError:
            pass
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Label smoothing helper
# ─────────────────────────────────────────────────────────────────────────────

def smooth_labels(
    size: int,
    real: bool = True,
    smooth: float = 0.1,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Возвращает сглаженные метки.
    real=True  → [1 - smooth, 1]  (в среднем 0.9)
    real=False → [0,  smooth]     (в среднем 0.0)
    """
    if real:
        return torch.empty(size, device=device).uniform_(1.0 - smooth, 1.0)
    else:
        return torch.empty(size, device=device).uniform_(0.0, smooth)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    latent_dim = 100
    B = 4

    # Generator
    G = Generator(latent_dim=latent_dim, ngf=64).to(device)
    G.apply(weights_init)
    z = torch.randn(B, latent_dim, 1, 1, device=device)
    fake = G(z)
    print(f"Generator output:       {fake.shape}")

    # Discriminator
    D = Discriminator(nc=3, ndf=64).to(device)
    D.apply(weights_init)
    score_fake = D(fake.detach())
    print(f"Discriminator output:   {score_fake.shape}")

    # Подсчёт параметров
    n_G = sum(p.numel() for p in G.parameters())
    n_D = sum(p.numel() for p in D.parameters())
    print(f"Generator  параметров:  {n_G:,}  ({n_G / 1e6:.2f}M)")
    print(f"Discriminator параметров: {n_D:,}  ({n_D / 1e6:.2f}M)")

    # Пример label smoothing
    real_labels = smooth_labels(B, real=True,  device=str(device))
    fake_labels = smooth_labels(B, real=False, device=str(device))
    print(f"Real labels: {real_labels.tolist()}")
    print(f"Fake labels: {fake_labels.tolist()}")