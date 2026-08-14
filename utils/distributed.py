"""Утилиты для инициализации и работы с DistributedDataParallel (DDP)."""

from __future__ import annotations

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def is_distributed() -> bool:
    """Проверяет, активен ли распределенный режим PyTorch DDP."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Возвращает глобальный ранг текущего процесса."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """Возвращает общее количество задействованных GPU."""
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    """Проверяет, является ли текущий процесс главным (Rank 0)."""
    return get_rank() == 0


def setup_distributed() -> tuple[torch.device, int]:
    """Инициализирует DDP и привязывает локальный процесс к конкретной GPU."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        torch.backends.cudnn.benchmark = True
        return device, local_rank

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device, 0


def cleanup_distributed() -> None:
    """Завершает процесс-группу DDP."""
    if is_distributed():
        dist.destroy_process_group()


def wrap_ddp(
    model: torch.nn.Module,
    local_rank: int,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    """Оборачивает модуль PyTorch в DDP."""
    if not is_distributed():
        return model

    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters,
    )


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Синхронизирует и усредняет скалярный тензор со всех GPU."""
    if not is_distributed():
        return tensor
    reduced = tensor.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= get_world_size()
    return reduced
