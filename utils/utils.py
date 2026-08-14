"""Вспомогательные утилиты для логирования, сохранения чекпоинтов и сэмплов."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torchvision.utils as vutils
import yaml

from utils.distributed import is_main_process


def count_parameters(model: torch.nn.Module) -> int:
    """Считает общее количество обучаемых параметров модели."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: torch.nn.Module, model_name: str = "Model") -> None:
    """Печатает информацию о модели только на главном процессе."""
    if not is_main_process():
        return
    total = count_parameters(model)
    print(f"\n{'═' * 52}")
    print(f"  {model_name}")
    print(f"  Обучаемых параметров: {total:,} ({total / 1e6:.2f}M)")
    print(f"{'═' * 52}\n")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Загружает YAML-конфигурацию."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(
    state: dict[str, Any],
    is_best: bool,
    checkpoint_dir: str | Path,
    filename: str = "latest.pt",
) -> None:
    """Сохраняет чекпоинт обучения без префикса module. только на Rank 0."""
    if not is_main_process():
        return

    save_dir = Path(checkpoint_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    raw_model = (
        state["model"].module
        if hasattr(state["model"], "module")
        else state["model"]
    )

    payload = {
        "epoch": state["epoch"],
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": state["optimizer"].state_dict(),
        "config": state.get("config", {}),
    }

    if "scaler" in state and state["scaler"] is not None:
        payload["scaler_state_dict"] = state["scaler"].state_dict()

    torch.save(payload, save_dir / filename)
    if is_best:
        torch.save(payload, save_dir / "best_model.pt")


def save_samples(
    images: torch.Tensor,
    output_path: str | Path,
    nrow: int = 8,
    normalize: bool = True,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> None:
    """Сохраняет сетку сгенерированных изображений на диск."""
    if not is_main_process():
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(
        images,
        path,
        nrow=nrow,
        normalize=normalize,
        value_range=value_range,
    )
