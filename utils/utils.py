"""
Утилиты для обучения VAE, GAN, DDPM.

Содержит:
  - TrainingLogger   — CSV + консольный лог метрик
  - Visualizer       — matplotlib графики (dark theme)
  - save_checkpoint / load_checkpoint
  - save_sample_grid — сетка изображений → PNG
  - denormalize      — [-1,1] → [0,1]
  - compare_models   — сравнительный постер всех моделей
  - count_parameters — подсчёт параметров модели
"""

import os
import csv
import time
import math
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision.utils import make_grid


# ─────────────────────────────────────────────────────────────────────────────
# Утилиты для тензоров / изображений
# ─────────────────────────────────────────────────────────────────────────────

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Переводит тензор из диапазона [-1, 1] в [0, 1]."""
    return (tensor.detach().cpu() + 1.0) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Чекпоинты
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    state: Dict,
    output_dir: str,
    filename: str = "checkpoint.pt",
    is_best: bool = False,
) -> str:
    """Сохраняет чекпоинт модели. Возвращает путь к файлу."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(output_dir, "best.pt")
        torch.save(state, best_path)
        print(f"  ✓ Новый лучший чекпоинт → {best_path}")
    return path


def load_checkpoint(path: str, device: str = "cpu") -> Dict:
    """Загружает чекпоинт с диска."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Чекпоинт не найден: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    print(f"  ✓ Чекпоинт загружен: {path}  (epoch={ckpt.get('epoch', '?')})")
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Сохранение сетки изображений
# ─────────────────────────────────────────────────────────────────────────────

def _apply_dark_style() -> None:
    """Применяет тёмную тему matplotlib."""
    plt.rcParams.update({
        "figure.facecolor":  "#0d1117",
        "axes.facecolor":    "#161b22",
        "axes.edgecolor":    "#30363d",
        "axes.labelcolor":   "#c9d1d9",
        "text.color":        "#c9d1d9",
        "xtick.color":       "#8b949e",
        "ytick.color":       "#8b949e",
        "grid.color":        "#21262d",
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "lines.linewidth":   2.0,
        "font.family":       "DejaVu Sans",
        "axes.titlecolor":   "#c9d1d9",
    })


def save_sample_grid(
    images: torch.Tensor,
    path: str,
    nrow: int = 8,
    title: str = "",
) -> None:
    """
    Сохраняет сетку изображений в PNG.

    Args:
        images: тензор (N, C, H, W) в диапазоне [-1, 1]
        path:   путь для сохранения
        nrow:   количество изображений в строке
        title:  заголовок сетки
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _apply_dark_style()

    imgs = denormalize(images).clamp(0, 1)
    nrow = min(nrow, len(imgs))
    ncol = math.ceil(len(imgs) / nrow)
    grid = make_grid(imgs, nrow=nrow, padding=2, pad_value=0.12)

    fig, ax = plt.subplots(figsize=(nrow * 1.6, ncol * 1.6 + 0.4))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, pad=4)
    fig.patch.set_facecolor("#0d1117")
    plt.tight_layout(pad=0.3)
    plt.savefig(path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Логгер метрик
# ─────────────────────────────────────────────────────────────────────────────

class TrainingLogger:
    """
    Логирует метрики обучения в CSV-файл и консоль.

    Пример использования:
        logger = TrainingLogger("checkpoints/vae", "VAE")
        logger.log(epoch=1, step=10, total_loss=0.5, recon=0.4, kl=0.1)
        logger.log_epoch(epoch=1, total_loss=0.5)
        logger.print_epoch_summary(epoch=1, total_loss=0.5)
    """

    def __init__(self, output_dir: str, model_name: str = "Model"):
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.output_dir / "training_log.csv"
        self._writer: Optional[csv.DictWriter] = None
        self._file = None
        self._fieldnames: List[str] = []

        # Полная история для графиков (по шагам)
        self.history: Dict[str, List[float]] = defaultdict(list)
        # История по эпохам (для финальных графиков)
        self.epoch_history: Dict[str, List[float]] = defaultdict(list)

        self._start_time = time.time()

    def log(self, epoch: int, step: int, **metrics) -> None:
        """Логирует один шаг в CSV."""
        elapsed = time.time() - self._start_time
        row = {"epoch": epoch, "step": step, "time_s": f"{elapsed:.1f}", **metrics}

        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames,
                                          extrasaction="ignore")
            self._writer.writeheader()

        for k in list(row.keys()):
            if k not in self._fieldnames:
                self._fieldnames.append(k)

        self._writer.writerow({k: row.get(k, "") for k in self._fieldnames})
        self._file.flush()

        for k, v in metrics.items():
            self.history[k].append(float(v))

    def log_epoch(self, epoch: int, **metrics) -> None:
        """Добавляет средние значения эпохи в epoch_history."""
        self.epoch_history["epoch"].append(epoch)
        for k, v in metrics.items():
            self.epoch_history[k].append(float(v))

    def print_step(
        self,
        epoch: int,
        total_epochs: int,
        step: int,
        total_steps: int,
        **metrics,
    ) -> None:
        """Выводит прогресс одного шага (без перевода строки)."""
        elapsed = time.time() - self._start_time
        metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(
            f"\r[{self.model_name}] "
            f"Epoch {epoch:3d}/{total_epochs} "
            f"[{step:4d}/{total_steps}]  "
            f"{metrics_str}  "
            f"({elapsed:.0f}s)",
            end="",
            flush=True,
        )

    def print_epoch_summary(self, epoch: int, **metrics) -> None:
        """Выводит итоги эпохи с разделителем."""
        elapsed = time.time() - self._start_time
        metrics_str = "  |  ".join(f"{k}: {v:.5f}" for k, v in metrics.items())
        print(f"\n{'─' * 72}")
        print(f"  Epoch {epoch:3d}  |  {metrics_str}  |  {elapsed:.0f}s total")
        print(f"{'─' * 72}")

    def close(self) -> None:
        if self._file:
            self._file.close()


# ─────────────────────────────────────────────────────────────────────────────
# Визуализатор
# ─────────────────────────────────────────────────────────────────────────────

# Цветовая палитра
_PALETTE = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff", "#ffa657", "#79c0ff", "#ff7b72"]


def _ema(values: List[float], alpha: float = 0.3) -> List[float]:
    """Экспоненциальное скользящее среднее."""
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result


class Visualizer:
    """
    Строит графики loss curves во время и после обучения.

    Поддерживает:
      - Сохранение в файл (каждые N эпох)
      - Inline-режим в Jupyter через IPython.display
    """

    def __init__(self, output_dir: str, model_name: str, inline: bool = False):
        """
        Args:
            output_dir: директория модели (plots/ создаётся внутри)
            model_name: 'VAE', 'GAN', 'DDPM'
            inline:     True → вывод в Jupyter через IPython.display
        """
        self.plots_dir = Path(output_dir) / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.inline = inline

        if inline:
            matplotlib.use("inline")
        else:
            matplotlib.use("Agg")

    def plot_curves(
        self,
        epoch_history: Dict[str, List[float]],
        epoch: int,
        save: bool = True,
        show: bool = False,
    ) -> Optional[str]:
        """
        Рисует loss curves по эпохам.

        Args:
            epoch_history: {"total_loss": [...], "recon_loss": [...], ...}
            epoch:         текущая эпоха (для имени файла)
            save:          сохранить PNG
            show:          показать inline (Jupyter)
        Returns:
            путь к сохранённому файлу или None
        """
        _apply_dark_style()

        # Убираем служебное поле epoch из отрисовки
        histories = {k: v for k, v in epoch_history.items() if k != "epoch"}
        if not histories:
            return None

        n = len(histories)
        cols = min(n, 3)
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
        if n == 1:
            axes = [axes]
        else:
            axes = np.array(axes).flatten().tolist()

        fig.suptitle(
            f"{self.model_name}  —  Training Curves  (Epoch {epoch})",
            fontsize=12,
            color="#58a6ff",
        )

        for idx, (name, values) in enumerate(histories.items()):
            if idx >= len(axes):
                break
            ax = axes[idx]
            color = _PALETTE[idx % len(_PALETTE)]
            xs = list(range(1, len(values) + 1))

            ax.plot(xs, values, color=color, alpha=0.35, linewidth=1.0, label="raw")
            if len(values) >= 5:
                ax.plot(xs, _ema(values, 0.25), color=color, linewidth=2.2, label="EMA")

            ax.set_title(name.replace("_", " ").title(), fontsize=10)
            ax.set_xlabel("Epoch", fontsize=8)
            ax.grid(True)
            ax.legend(fontsize=7, framealpha=0.2)

        # Скрываем пустые оси
        for ax in axes[n:]:
            ax.set_visible(False)

        plt.tight_layout()

        save_path = None
        if save:
            save_path = str(self.plots_dir / f"epoch_{epoch:04d}.png")
            plt.savefig(save_path, dpi=100, bbox_inches="tight",
                        facecolor=fig.get_facecolor())

        if show:
            try:
                from IPython.display import display, clear_output
                clear_output(wait=True)
                display(fig)
            except ImportError:
                pass

        plt.close(fig)
        return save_path

    def plot_final_summary(
        self,
        epoch_history: Dict[str, List[float]],
        samples: Optional[torch.Tensor] = None,
        extra_info: str = "",
    ) -> str:
        """
        Итоговый постер: loss curves слева + сетка образцов справа.

        Returns:
            путь к сохранённому файлу
        """
        _apply_dark_style()

        histories = {k: v for k, v in epoch_history.items() if k != "epoch"}
        has_samples = samples is not None
        n = len(histories)

        if has_samples:
            fig = plt.figure(figsize=(18, 10))
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.4], wspace=0.08)
            n_sub_cols = min(2, n)
            n_sub_rows = math.ceil(n / n_sub_cols)
            gs_left = gridspec.GridSpecFromSubplotSpec(
                n_sub_rows, n_sub_cols, subplot_spec=gs[0], hspace=0.55, wspace=0.4
            )
        else:
            cols = min(n, 3)
            rows = math.ceil(n / cols)
            fig, axes_flat = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
            if n == 1:
                axes_flat = [axes_flat]
            else:
                axes_flat = np.array(axes_flat).flatten().tolist()

        for idx, (name, values) in enumerate(histories.items()):
            if has_samples:
                r, c = divmod(idx, n_sub_cols)
                ax = fig.add_subplot(gs_left[r, c])
            else:
                if idx >= len(axes_flat):
                    break
                ax = axes_flat[idx]

            color = _PALETTE[idx % len(_PALETTE)]
            xs = list(range(1, len(values) + 1))
            ax.plot(xs, values, color=color, alpha=0.3, linewidth=1.0)
            if len(values) >= 5:
                ax.plot(xs, _ema(values, 0.2), color=color, linewidth=2.5)
            ax.set_title(name.replace("_", " ").title(), fontsize=9)
            ax.set_xlabel("Epoch", fontsize=7)
            ax.grid(True)
            ax.tick_params(labelsize=7)

        if has_samples:
            ax_img = fig.add_subplot(gs[1])
            imgs = denormalize(samples).clamp(0, 1)
            nrow = min(8, len(imgs))
            grid = make_grid(imgs, nrow=nrow, padding=2, pad_value=0.12)
            ax_img.imshow(grid.permute(1, 2, 0).numpy())
            ax_img.axis("off")
            ax_img.set_title(
                f"Generated Samples  (n={len(imgs)})", fontsize=10, pad=8, color="#3fb950"
            )
        elif n < len(axes_flat):
            for ax in axes_flat[n:]:
                ax.set_visible(False)

        epochs_done = len(next(iter(histories.values()))) if histories else "?"
        fig.suptitle(
            f"✦  {self.model_name}  —  Final Results  "
            f"(Epochs: {epochs_done})  {extra_info}",
            fontsize=14,
            color="#58a6ff",
            y=1.02,
        )

        save_path = str(self.plots_dir / "final_summary.png")
        plt.savefig(save_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓ Финальный постер → {save_path}")
        return save_path


# ─────────────────────────────────────────────────────────────────────────────
# Сравнение всех моделей
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(
    results: Dict[str, Dict],
    save_path: str = "checkpoints/comparison.png",
) -> None:
    """
    Строит сравнительный постер VAE / GAN / DDPM бок о бок.

    Args:
        results: {
            "VAE":  {"samples": Tensor(N,C,H,W), "loss_history": [...]},
            "GAN":  {"samples": Tensor,          "loss_history": [...]},
            "DDPM": {"samples": Tensor,          "loss_history": [...]},
        }
        save_path: путь для сохранения PNG
    """
    _apply_dark_style()

    models = list(results.keys())
    n = len(models)
    fig, axes = plt.subplots(2, n, figsize=(n * 7, 12))
    if n == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        "✦  Сравнение генеративных моделей: VAE / GAN / DDPM",
        fontsize=16,
        color="#58a6ff",
        y=1.01,
    )

    for col, name in enumerate(models):
        data = results[name]
        color = _PALETTE[col % len(_PALETTE)]

        # ── Верх: loss curve ─────────────────────────────────────────────────
        ax_loss = axes[0, col]
        loss_hist = data.get("loss_history", [])
        if loss_hist:
            xs = list(range(1, len(loss_hist) + 1))
            ax_loss.plot(xs, loss_hist, color=color, alpha=0.35, lw=1.0)
            if len(loss_hist) >= 5:
                ax_loss.plot(xs, _ema(loss_hist, 0.2), color=color, lw=2.5)
        ax_loss.set_title(f"{name} — Loss", fontsize=11)
        ax_loss.set_xlabel("Epoch", fontsize=9)
        ax_loss.grid(True)

        # ── Низ: образцы ─────────────────────────────────────────────────────
        ax_img = axes[1, col]
        samples = data.get("samples")
        if samples is not None:
            imgs = denormalize(samples).clamp(0, 1)
            grid = make_grid(imgs[:16], nrow=4, padding=2, pad_value=0.12)
            ax_img.imshow(grid.permute(1, 2, 0).numpy())
        else:
            ax_img.text(0.5, 0.5, "Нет данных", ha="center", va="center",
                        transform=ax_img.transAxes, fontsize=12)
        ax_img.axis("off")
        ax_img.set_title("Сгенерированные образцы", fontsize=10)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Сравнительный постер → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Подсчёт параметров
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> int:
    """Возвращает количество обучаемых параметров модели."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: torch.nn.Module, model_name: str = "Model") -> None:
    """Выводит имя модели и количество параметров."""
    total = count_parameters(model)
    print(f"\n{'═' * 52}")
    print(f"  {model_name}")
    print(f"  Параметров: {total:,}  ({total / 1e6:.2f}M)")
    print(f"{'═' * 52}\n")
