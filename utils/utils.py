"""
Общие утилиты обучения.
"""

from __future__ import annotations

import csv
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

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
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Конфигурация не найдена: {path}"
        )

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
    if key in config:
        return config[key]

    aliases = {
        "data_root": ("data", "root"),
        "output_dir": ("logging", "output_dir"),
    }

    if key in aliases:
        section_name, section_key = aliases[key]

        section = config.get(
            section_name,
            {},
        )

        if isinstance(section, dict):
            if section_key in section:
                return section[section_key]

    for section in config.values():
        if not isinstance(section, dict):
            continue

        if key in section:
            return section[key]

    return default


def count_parameters(
    model: torch.nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def print_model_info(
    model: torch.nn.Module,
    model_name: str = "Model",
) -> None:
    if not is_main_process():
        return

    model = unwrap_model(model)

    total = count_parameters(model)

    print()
    print("=" * 64)
    print(model_name)
    print(
        f"Обучаемых параметров: "
        f"{total:,} "
        f"({total / 1e6:.2f}M)"
    )
    print("=" * 64)
    print()


def denormalize(
    tensor: torch.Tensor,
) -> torch.Tensor:
    return (
        tensor.detach().float().cpu()
        + 1.0
    ) / 2.0


def save_sample_grid(
    images: torch.Tensor,
    path: str | Path,
    nrow: int = 8,
    title: str = "",
) -> None:
    if not is_main_process():
        return

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = denormalize(images).clamp(
        0.0,
        1.0,
    )

    vutils.save_image(
        images,
        path,
        nrow=nrow,
    )

    if title:
        print(
            f"Saved samples: {path}"
        )


def save_samples(
    images: torch.Tensor,
    output_path: str | Path,
    nrow: int = 8,
    normalize: bool = True,
    value_range: tuple[float, float] = (
        -1.0,
        1.0,
    ),
) -> None:
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


def _state_dict_if_module(
    value: Any,
) -> Any:
    if isinstance(value, torch.nn.Module):
        return unwrap_model(value).state_dict()

    return value


def _state_dict_if_object(
    value: Any,
) -> Any:
    if hasattr(value, "state_dict"):
        return value.state_dict()

    return value


def save_checkpoint(
    state: dict[str, Any],
    output_dir: str | Path | None = None,
    filename: str = "checkpoint.pt",
    is_best: bool = False,
    checkpoint_dir: str | Path | None = None,
) -> str:
    if not is_main_process():
        return ""

    directory = (
        checkpoint_dir
        if checkpoint_dir is not None
        else output_dir
    )

    if directory is None:
        raise ValueError(
            "Не указан output_dir."
        )

    directory = Path(directory)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = dict(state)

    module_keys = (
        "model",
        "model_g",
        "model_d",
        "generator",
        "discriminator",
    )

    optimizer_keys = (
        "optimizer",
        "optimizer_g",
        "optimizer_d",
        "opt_g",
        "opt_d",
        "scheduler",
        "scaler",
        "scaler_g",
        "scaler_d",
    )

    for key in module_keys:
        if key in payload:
            payload[key] = _state_dict_if_module(
                payload[key]
            )

    for key in optimizer_keys:
        if key in payload:
            payload[key] = _state_dict_if_object(
                payload[key]
            )

    path = directory / filename

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
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint не найден: {path}"
        )

    return torch.load(
        path,
        map_location=device,
        weights_only=False,
    )


class TrainingLogger:
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

        self.csv_path = (
            self.output_dir
            / "training_log.csv"
        )

        self._file = None
        self._writer = None
        self._fieldnames: list[str] = []

        self.history = defaultdict(list)
        self.epoch_history = defaultdict(list)

        self._start_time = time.time()

    def log(
        self,
        epoch: int,
        step: int,
        **metrics: float,
    ) -> None:
        row = {
            "epoch": epoch,
            "step": step,
            "time_s": round(
                time.time()
                - self._start_time,
                3,
            ),
            **metrics,
        }

        if self._writer is None:
            self._fieldnames = list(
                row.keys()
            )

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

        self._writer.writerow(row)
        self._file.flush()

        for key, value in metrics.items():
            self.history[key].append(
                float(value)
            )

    def log_epoch(
        self,
        epoch: int,
        **metrics: float,
    ) -> None:
        self.epoch_history[
            "epoch"
        ].append(epoch)

        for key, value in metrics.items():
            self.epoch_history[
                key
            ].append(float(value))

    def print_epoch_summary(
        self,
        epoch: int,
        **metrics: float,
    ) -> None:
        values = " | ".join(
            f"{key}: {value:.5f}"
            for key, value in metrics.items()
        )

        print(
            f"Epoch {epoch:04d} | {values}"
        )

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class Visualizer:
    def __init__(
        self,
        output_dir: str | Path,
        model_name: str,
        inline: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.inline = inline

        self.plots_dir = (
            self.output_dir
            / "plots"
        )

        self.plots_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def plot_curves(
        self,
        epoch_history: dict,
        epoch: int,
        save: bool = True,
        show: bool = False,
    ) -> str | None:
        histories = {
            key: value
            for key, value
            in epoch_history.items()
            if key != "epoch"
            and value
        }

        if not histories:
            return None

        columns = min(
            3,
            len(histories),
        )

        rows = math.ceil(
            len(histories)
            / columns
        )

        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(
                columns * 5,
                rows * 3,
            ),
        )

        axes = np.atleast_1d(
            axes
        ).reshape(-1)

        for index, (
            name,
            values,
        ) in enumerate(
            histories.items()
        ):
            axes[index].plot(values)
            axes[index].set_title(
                name
            )
            axes[index].set_xlabel(
                "Epoch"
            )
            axes[index].grid(
                True,
                alpha=0.3,
            )

        for axis in axes[
            len(histories):
        ]:
            axis.set_visible(False)

        fig.suptitle(
            f"{self.model_name} "
            f"— Epoch {epoch}"
        )

        fig.tight_layout()

        output = None

        if save:
            output = str(
                self.plots_dir
                / f"epoch_{epoch:04d}.png"
            )

            fig.savefig(
                output,
                dpi=120,
                bbox_inches="tight",
            )

        if show:
            plt.show()

        plt.close(fig)

        return output
