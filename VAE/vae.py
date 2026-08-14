"""
β-VAE (Variational Autoencoder) для изображений 128×128×3.

Улучшения по сравнению с базовой версией:
  - ResidualBlock  — стабилизирует обучение глубоких сетей
  - SelfAttention2d — захватывает глобальные зависимости (на 8×8)
  - Параметр beta  — β-VAE (Higgins et al., 2017):
        beta=1 → обычный VAE
        beta>1 → более диsentangled латентное пространство
  - Loss нормализована по пикселям (не по батчу)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Строительные блоки
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Residual блок: Conv→GroupNorm→SiLU→Conv→GroupNorm + skip connection.
    Если in_channels ≠ out_channels — добавляет 1×1 conv для выравнивания.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = min(8, out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(groups, out_channels),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class SelfAttention2d(nn.Module):
    """
    Multi-head Self-Attention для 2D feature maps.
    Применяется на малых разрешениях (8×8, 16×16) для захвата
    глобальных зависимостей между регионами изображения.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, \
            f"channels ({channels}) должен делиться на num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.scale     = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv  = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        x = self.norm(x)

        # Проекции Q, K, V
        qkv = self.qkv(x)                          # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)              # каждый (B, C, H, W)

        def to_heads(t: torch.Tensor) -> torch.Tensor:
            # (B, C, H, W) → (B, heads, H*W, head_dim)
            t = t.view(B, self.num_heads, self.head_dim, H * W)
            return t.permute(0, 1, 3, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        # Attention: (B, heads, H*W, H*W)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Взвешенная сумма значений
        out = torch.matmul(attn, v)                 # (B, heads, H*W, head_dim)
        out = out.permute(0, 1, 3, 2).contiguous()  # (B, heads, head_dim, H*W)
        out = out.view(B, C, H, W)

        return self.proj(out) + residual


# ─────────────────────────────────────────────────────────────────────────────
# Энкодер / Декодер
# ─────────────────────────────────────────────────────────────────────────────

def _enc_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Энкодер-блок: DownConv → ResBlock."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(),
        ResidualBlock(out_ch, out_ch),
    )


def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Декодер-блок: ResBlock → UpConv."""
    return nn.Sequential(
        ResidualBlock(in_ch, in_ch),
        nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# β-VAE
# ─────────────────────────────────────────────────────────────────────────────

class VAE(nn.Module):
    """
    β-VAE для изображений 128×128×3.

    Архитектура энкодера:
        128×128×3
        → 64×64×64   (enc1)
        → 32×32×128  (enc2)
        → 16×16×256  (enc3)
        →  8×8×512   (enc4 + SelfAttention)
        →  4×4×512   (enc5)
        →  2×2×512   (enc6)
        → fc_mu / fc_logvar → z ∈ R^latent_dim

    Декодер — зеркальная структура.
    """

    def __init__(self, latent_dim: int = 256, beta: float = 4.0):
        """
        Args:
            latent_dim: размерность латентного пространства
            beta:       коэффициент KL-потерь (β=1 → vanilla VAE)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.beta       = beta

        # ── Энкодер ──────────────────────────────────────────────────────────
        self.enc1 = _enc_block(3,   64)    # 128 → 64
        self.enc2 = _enc_block(64,  128)   # 64  → 32
        self.enc3 = _enc_block(128, 256)   # 32  → 16
        self.enc4 = _enc_block(256, 512)   # 16  → 8
        self.enc4_attn = SelfAttention2d(512, num_heads=4)   # attention на 8×8
        self.enc5 = _enc_block(512, 512)   # 8   → 4
        self.enc6 = _enc_block(512, 512)   # 4   → 2

        # Bottleneck → μ, log σ²
        self.fc_mu     = nn.Linear(512 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(512 * 2 * 2, latent_dim)

        # ── Декодер ──────────────────────────────────────────────────────────
        self.decoder_fc = nn.Linear(latent_dim, 512 * 2 * 2)

        self.dec6 = _dec_block(512, 512)   # 2  → 4
        self.dec5 = _dec_block(512, 512)   # 4  → 8
        self.dec4 = _dec_block(512, 256)   # 8  → 16
        self.dec4_attn = SelfAttention2d(256, num_heads=4)   # attention на 16×16
        self.dec3 = _dec_block(256, 128)   # 16 → 32
        self.dec2 = _dec_block(128, 64)    # 32 → 64

        # Финальный апсэмплинг 64 → 128
        self.dec1 = nn.Sequential(
            ResidualBlock(64, 64),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    # ── Encode / Reparameterize / Decode ─────────────────────────────────────

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Прогоняет x через энкодер.
        Returns:
            mu:     (B, latent_dim)
            logvar: (B, latent_dim)
        """
        h = self.enc1(x)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.enc4(h)
        h = self.enc4_attn(h)
        h = self.enc5(h)
        h = self.enc6(h)
        h = h.flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Трюк репараметризации: z = μ + σ·ε, ε ~ N(0, I).
        Во время инференса (eval mode) возвращает μ.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Декодирует латентный вектор z в изображение."""
        h = self.decoder_fc(z)
        h = h.view(-1, 512, 2, 2)
        h = self.dec6(h)
        h = self.dec5(h)
        h = self.dec4(h)
        h = self.dec4_attn(h)
        h = self.dec3(h)
        h = self.dec2(h)
        return self.dec1(h)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            recon:  восстановленное изображение (B, 3, 128, 128)
            mu:     среднее q(z|x)
            logvar: log-дисперсия q(z|x)
        """
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decode(z)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, n: int = 16, device: str = "cpu") -> torch.Tensor:
        """Генерирует n случайных изображений из p(z) = N(0, I)."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)


# ─────────────────────────────────────────────────────────────────────────────
# Функция потерь
# ─────────────────────────────────────────────────────────────────────────────

def vae_loss(
    recon_x: torch.Tensor,
    x:       torch.Tensor,
    mu:      torch.Tensor,
    logvar:  torch.Tensor,
    beta:    float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    β-VAE Loss = Reconstruction Loss + β · KL Divergence.

    Нормализована по количеству элементов (не sum, а mean по пикселям).

    Args:
        recon_x: восстановленное изображение (B, C, H, W)
        x:       оригинальное изображение    (B, C, H, W)
        mu:      μ из энкодера               (B, latent_dim)
        logvar:  log σ² из энкодера          (B, latent_dim)
        beta:    коэффициент KL

    Returns:
        total_loss, recon_loss, kl_loss  — все как скаляры
    """
    # Reconstruction: MSE по пикселям (нормализована)
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")

    # KL divergence: -0.5 * Σ(1 + log σ² - μ² - σ²)
    # нормализована по batch_size и latent_dim
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            device = torch.nn.DataParallel(device)
    else:
        print(f"Device: {device}")

    model = VAE(latent_dim=256, beta=4.0).to(device)

    x = torch.randn(2, 3, 128, 128, device=device)
    recon, mu, logvar = model(x)

    total, recon_l, kl_l = vae_loss(recon, x, mu, logvar, beta=model.beta)

    print(f"Input:     {x.shape}")
    print(f"Recon:     {recon.shape}")
    print(f"Mu:        {mu.shape}")
    print(f"Logvar:    {logvar.shape}")
    print(f"Loss:      total={total.item():.4f}  recon={recon_l.item():.4f}  kl={kl_l.item():.4f}")

    # Проверка генерации
    samples = model.sample(n=4, device=str(device))
    print(f"Samples:   {samples.shape}  min={samples.min():.3f}  max={samples.max():.3f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров: {total_params:,}  ({total_params / 1e6:.2f}M)")