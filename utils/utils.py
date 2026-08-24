"""
Общие утилиты обучения VAE, GAN и DDPM.

Поддерживает:

- конфигурации YAML;
- TrainingLogger;
- Visualizer;
- checkpoint save/load;
- sample grids;
- совместимость с DDP;
- совместимость со старым API.
"""

from __future__ import annotations

import csv
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
import yaml

from utils.distributed import (
    is_main_process,
    unwrap_model,
)


def load_config(
    config_path: str | Path,
) -> dict[str, Any]:
    """Загружает YAML."""

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Конфигурация не найдена: {path}")

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file) or {}


def get_config_value(
    config: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """
    Ищет параметр как в старом плоском YAML,
    так и в новом секционном YAML.

    Например:

        training:
          batch_size: 16

    будет доступен как:

        get_config_value(
            config,
            "batch_size",
        )
    """

    if key in config:
        return config[key]

    aliases = {
        "data_root": (
            "data",
            "root",
        ),
        "output_dir": (
            "logging",
            "output_dir",
        ),
    }

    if key in aliases:
        section_name, section_key = aliases[key]

        section = config.get(
            section_name,
            {},
        )

        if isinstance(
            section,
            dict,
        ):
            if section_key in section:
                return section[section_key]

    for section in config.values():
        if not isinstance(
            section,
            dict,
        ):
            continue

        if key in section:
            return section[key]

    return default


def count_parameters(
    model: torch.nn.Module,
) -> int:
    """Количество обучаемых параметров."""

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def print_model_info(
    model: torch.nn.Module,
    model_name: str = "Model",
) -> None:
    """Печатает информацию только на rank 0."""

    if not is_main_process():
        return

    model = unwrap_model(model)

    total = count_parameters(model)

    print()
    print("=" * 64)
    print(model_name)
    print(f"Обучаемых параметров: {total:,} ({total / 1e6:.2f}M)")
    print("=" * 64)
    print()


def denormalize(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """[-1, 1] -> [0, 1]."""

    return (tensor.detach().cpu() + 1.0) / 2.0


def save_sample_grid(
    images: torch.Tensor,
    path: str | Path,
    nrow: int = 8,
    title: str = "",
) -> None:
    """Сохраняет grid изображений."""

    if not is_main_process():
        return

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = denormalize(images).clamp(0.0, 1.0)

    vutils.save_image(
        images,
        path,
        nrow=nrow,
    )

    if title:
        # Заголовок опциональный.
        # Сам PNG уже сохранён выше.
        pass


def compare_models(
    results: dict[str, dict],
    save_path: str | Path,
    nrow: int = 8,
) -> Path | None:
    """
    Сохраняет сравнительный постер из семплов нескольких моделей.

    Args:
        results: {имя_модели: {"samples": tensor(B,C,H,W), ...}}
        save_path: путь для сохранения PNG.
        nrow: изображений в строке внутри сетки каждой модели.

    Returns:
        Путь к сохранённому файлу или None (не main process / нет данных).
    """

    if not is_main_process():
        return None

    entries = {
        name: content["samples"]
        for name, content in results.items()
        if isinstance(content, dict) and content.get("samples") is not None
    }

    if not entries:
        return None

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_models = len(entries)

    fig, axes = plt.subplots(
        n_models,
        1,
        figsize=(10, 4 * n_models),
    )

    axes = np.atleast_1d(axes).reshape(-1)

    for ax, (name, samples) in zip(axes, entries.items()):
        grid = vutils.make_grid(
            denormalize(samples).clamp(0.0, 1.0).cpu(),
            nrow=nrow,
            padding=2,
        )

        ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.set_title(name, fontsize=14)
        ax.axis("off")

    for ax in axes[len(entries) :]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return path


def save_samples(
    images: torch.Tensor,
    output_path: str | Path,
    nrow: int = 8,
    normalize: bool = True,
    value_range: tuple[
        float,
        float,
    ] = (-1.0, 1.0),
) -> None:
    """Совместимый API для новых train.py."""

    if not is_main_process():
        return

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    kwargs = {
        "nrow": nrow,
        "normalize": normalize,
    }

    if normalize:
        kwargs["value_range"] = value_range

    vutils.save_image(
        images.detach().cpu(),
        path,
        **kwargs,
    )


def save_checkpoint(
    state: dict[str, Any],
    output_dir: str | Path | None = None,
    filename: str = "checkpoint.pt",
    is_best: bool = False,
    checkpoint_dir: str | Path | None = None,
) -> str:
    """
    Сохраняет checkpoint.

    Поддерживает старый API:

        save_checkpoint(
            state,
            output_dir,
            filename,
            is_best,
        )

    и новый:

        save_checkpoint(
            state,
            is_best=False,
            checkpoint_dir=...,
        )
    """

    if not is_main_process():
        return ""

    directory = checkpoint_dir if checkpoint_dir is not None else output_dir

    if directory is None:
        raise ValueError("Не указан output_dir/checkpoint_dir.")

    directory = Path(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = directory / filename

    payload = dict(state)

    # Старый API:
    # state["model"] = nn.Module
    if isinstance(
        payload.get("model"),
        torch.nn.Module,
    ):
        payload["model"] = unwrap_model(payload["model"]).state_dict()

    # Старый GAN API может использовать
    # model_g / model_d.
    for key in (
        "model_g",
        "model_d",
        "generator",
        "discriminator",
    ):
        value = payload.get(key)

        if isinstance(
            value,
            torch.nn.Module,
        ):
            payload[key] = unwrap_model(value).state_dict()

    # Optimizer objects -> state_dict.
    for key in (
        "optimizer",
        "optimizer_g",
        "optimizer_d",
        "opt_g",
        "opt_d",
        "scheduler",
    ):
        value = payload.get(key)

        if hasattr(
            value,
            "state_dict",
        ):
            payload[key] = value.state_dict()

    scaler = payload.get("scaler")

    if hasattr(
        scaler,
        "state_dict",
    ):
        payload["scaler"] = scaler.state_dict()

    scaler_g = payload.get("scaler_g")

    if hasattr(
        scaler_g,
        "state_dict",
    ):
        payload["scaler_g"] = scaler_g.state_dict()

    scaler_d = payload.get("scaler_d")

    if hasattr(
        scaler_d,
        "state_dict",
    ):
        payload["scaler_d"] = scaler_d.state_dict()

    torch.save(
        payload,
        path,
    )

    if is_best:
        torch.save(
            payload,
            directory / "best.pt",
        )

    return str(path)


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Загружает checkpoint."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {path}")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    print(f"Checkpoint loaded: {path} (epoch={checkpoint.get('epoch', '?')})")

    return checkpoint


class TrainingLogger:
    """CSV + консольный лог."""

    def __init__(
        self,
        output_dir: str | Path,
        model_name: str = "Model",
    ):
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model_name = model_name

        self.csv_path = self.output_dir / "training_log.csv"

        self._file = None
        self._writer = None
        self._fieldnames = []

        self.history = defaultdict(list)

        self.epoch_history = defaultdict(list)

        self._start_time = time.time()

    def log(
        self,
        epoch: int,
        step: int,
        **metrics: float,
    ) -> None:
        elapsed = time.time() - self._start_time

        row = {
            "epoch": epoch,
            "step": step,
            "time_s": f"{elapsed:.1f}",
            **metrics,
        }

        if self._writer is None:
            self._fieldnames = list(row.keys())

            self._file = open(
                self.csv_path,
                "w",
                newline="",
                encoding="utf-8",
            )

            self._writer = csv.DictWriter(
                self._file,
                fieldnames=self._fieldnames,
                extrasaction="ignore",
            )

            self._writer.writeheader()

        self._writer.writerow(
            {
                key: row.get(
                    key,
                    "",
                )
                for key in self._fieldnames
            }
        )

        self._file.flush()

        for key, value in metrics.items():
            self.history[key].append(float(value))

    def log_epoch(
        self,
        epoch: int,
        **metrics: float,
    ) -> None:
        self.epoch_history["epoch"].append(epoch)

        for key, value in metrics.items():
            self.epoch_history[key].append(float(value))

    def print_epoch_summary(
        self,
        epoch: int,
        **metrics: float,
    ) -> None:
        elapsed = time.time() - self._start_time

        metrics_str = " | ".join(
            f"{key}: {value:.5f}" for key, value in metrics.items()
        )

        print()
        print("-" * 72)
        print(f"Epoch {epoch:04d} | {metrics_str} | {elapsed:.0f}s")
        print("-" * 72)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


class Visualizer:
    """Простая визуализация training curves."""

    def __init__(
        self,
        output_dir: str | Path,
        model_name: str,
        inline: bool = False,
    ):
        self.output_dir = Path(output_dir)

        self.plots_dir = self.output_dir / "plots"

        self.plots_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model_name = model_name
        self.inline = inline

        if not inline:
            matplotlib.use("Agg")

    def plot_curves(
        self,
        epoch_history: dict,
        epoch: int,
        save: bool = True,
        show: bool = False,
    ) -> str | None:
        histories = {
            key: value for key, value in epoch_history.items() if key != "epoch"
        }

        if not histories:
            return None

        columns = min(
            3,
            len(histories),
        )

        rows = math.ceil(len(histories) / columns)

        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(
                columns * 5,
                rows * 3,
            ),
        )

        axes = np.atleast_1d(axes).reshape(-1)

        for index, (
            name,
            values,
        ) in enumerate(histories.items()):
            ax = axes[index]

            ax.plot(
                range(
                    1,
                    len(values) + 1,
                ),
                values,
            )

            ax.set_title(
                name.replace(
                    "_",
                    " ",
                )
            )

            ax.set_xlabel("Epoch")

            ax.grid(
                True,
                alpha=0.3,
            )

        for ax in axes[len(histories) :]:
            ax.set_visible(False)

        fig.suptitle(f"{self.model_name} — Epoch {epoch}")

        fig.tight_layout()

        output = None

        if save:
            output = str(self.plots_dir / f"epoch_{epoch:04d}.png")

            fig.savefig(
                output,
                dpi=100,
                bbox_inches="tight",
            )

        if show:
            plt.show()

        plt.close(fig)

        return output
