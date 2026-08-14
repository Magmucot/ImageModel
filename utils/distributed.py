"""
Утилиты для DistributedDataParallel.

Поддерживает:

    python VAE/train.py --config configs/vae.yaml

и:

    torchrun --standalone \
        --nproc_per_node=2 \
        VAE/train.py \
        --config configs/vae.yaml

В single-GPU режиме DDP не используется.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import (
    DistributedDataParallel,
)


def is_distributed() -> bool:
    """Проверяет, инициализирован ли DDP."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Количество процессов."""
    if not is_distributed():
        return 1

    return dist.get_world_size()


def get_rank() -> int:
    """Global rank текущего процесса."""
    if not is_distributed():
        return 0

    return dist.get_rank()


def is_main_process() -> bool:
    """True только для rank 0."""
    return get_rank() == 0


def setup_distributed(
    device_arg: str = "auto",
) -> tuple[
    torch.device,
    int,
    int,
    bool,
]:
    """
    Инициализирует DDP.

    Returns:
        device:
            GPU/CPU текущего процесса.

        local_rank:
            GPU index на текущей машине.

        rank:
            Global rank.

        distributed:
            Используется ли DDP.
    """

    distributed = (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
    )

    if not distributed:
        if device_arg == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device_arg)

        return (
            device,
            0,
            0,
            False,
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "DDP запущен, но CUDA недоступна."
        )

    rank = int(
        os.environ["RANK"]
    )

    local_rank = int(
        os.environ["LOCAL_RANK"]
    )

    world_size = int(
        os.environ["WORLD_SIZE"]
    )

    torch.cuda.set_device(
        local_rank
    )

    device = torch.device(
        "cuda",
        local_rank,
    )

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    torch.backends.cudnn.benchmark = True

    return (
        device,
        local_rank,
        rank,
        True,
    )


def cleanup_distributed() -> None:
    """Корректно завершает DDP."""
    if is_distributed():
        dist.destroy_process_group()


def barrier() -> None:
    """Синхронизирует все процессы."""
    if is_distributed():
        dist.barrier()


def seed_everything(
    seed: int,
    rank: int = 0,
) -> None:
    """
    Устанавливает воспроизводимый seed.

    Каждый rank получает отдельную RNG sequence.
    """

    final_seed = seed + rank

    random.seed(final_seed)
    np.random.seed(final_seed)

    torch.manual_seed(
        final_seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            final_seed
        )


def wrap_ddp(
    model: torch.nn.Module,
    device: torch.device,
    local_rank: int,
    distributed: bool,
) -> torch.nn.Module:
    """Перемещает модель на GPU и при необходимости оборачивает в DDP."""

    model = model.to(device)

    if not distributed:
        return model

    return DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )


def unwrap_model(
    model: torch.nn.Module,
) -> torch.nn.Module:
    """Возвращает настоящую модель из DDP/DataParallel."""

    if isinstance(
        model,
        DistributedDataParallel,
    ):
        return model.module

    if isinstance(
        model,
        torch.nn.DataParallel,
    ):
        return model.module

    return model


def reduce_mean(
    value: torch.Tensor,
) -> torch.Tensor:
    """Усредняет scalar tensor между всеми GPU."""

    if not is_distributed():
        return value

    result = value.detach().clone()

    dist.all_reduce(
        result,
        op=dist.ReduceOp.SUM,
    )

    result /= get_world_size()

    return result


def reduce_tensor(
    value: torch.Tensor,
) -> torch.Tensor:
    """Alias для reduce_mean."""

    return reduce_mean(value)


def reduce_sum(
    value: torch.Tensor,
) -> torch.Tensor:
    """Суммирует scalar tensor между GPU."""

    if not is_distributed():
        return value

    result = value.detach().clone()

    dist.all_reduce(
        result,
        op=dist.ReduceOp.SUM,
    )

    return result


def broadcast_object(
    obj: Any,
    src: int = 0,
) -> Any:
    """Рассылает Python object всем rank."""

    if not is_distributed():
        return obj

    objects = [obj]

    dist.broadcast_object_list(
        objects,
        src=src,
    )

    return objects[0]