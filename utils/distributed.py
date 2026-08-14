"""
Утилиты для DistributedDataParallel.

Поддерживает:
    python VAE/train.py ...
    torchrun --nproc_per_node=4 VAE/train.py ...

В single-GPU режиме DDP не используется.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def is_distributed() -> bool:
    """Возвращает True, если запущен distributed process group."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Количество процессов/GPU."""
    if not is_distributed():
        return 1

    return dist.get_world_size()


def get_rank() -> int:
    """Глобальный rank текущего процесса."""
    if not is_distributed():
        return 0

    return dist.get_rank()


def is_main_process() -> bool:
    """True только для rank 0."""
    return get_rank() == 0


def setup_distributed(
    device_arg: str = "auto",
) -> tuple[torch.device, int, int, bool]:
    """
    Инициализирует DDP.

    Returns:
        device:
            torch.device текущего процесса.

        local_rank:
            GPU index внутри текущей машины.

        rank:
            global rank.

        distributed:
            используется ли DDP.
    """

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if not distributed:
        if device_arg == "auto":
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            device = torch.device(device_arg)

        return device, 0, 0, False

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if not torch.cuda.is_available():
        raise RuntimeError(
            "DDP mode запущен, но CUDA недоступна."
        )

    torch.cuda.set_device(local_rank)

    device = torch.device(
        "cuda",
        local_rank,
    )

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    return device, local_rank, rank, True


def cleanup_distributed() -> None:
    """Корректно завершает process group."""
    if is_distributed():
        dist.destroy_process_group()


def barrier() -> None:
    """Синхронизация всех процессов."""
    if is_distributed():
        dist.barrier()


def seed_everything(
    seed: int,
    rank: int = 0,
) -> None:
    """
    Устанавливает seed.

    Разные rank получают разные RNG sequence,
    но обучение остаётся воспроизводимым.
    """

    final_seed = seed + rank

    random.seed(final_seed)
    np.random.seed(final_seed)

    torch.manual_seed(final_seed)
    torch.cuda.manual_seed_all(final_seed)


def wrap_ddp(
    model: torch.nn.Module,
    device: torch.device,
    local_rank: int,
    distributed: bool,
) -> torch.nn.Module:
    """Оборачивает модель в DDP только при необходимости."""

    model = model.to(device)

    if not distributed:
        return model

    return DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )


def unwrap_model(
    model: torch.nn.Module,
) -> torch.nn.Module:
    """
    Возвращает настоящую модель из DDP/DataParallel.

    Используется для:
        state_dict
        EMA
        sample
        checkpoint
    """

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
    """
    Усредняет scalar tensor между всеми GPU.
    """

    if not is_distributed():
        return value

    result = value.detach().clone()

    dist.all_reduce(
        result,
        op=dist.ReduceOp.SUM,
    )

    result /= get_world_size()

    return result


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
    obj,
    src: int = 0,
):
    """Рассылает Python object всем rank."""

    if not is_distributed():
        return obj

    objects = [obj]

    dist.broadcast_object_list(
        objects,
        src=src,
    )

    return objects[0]