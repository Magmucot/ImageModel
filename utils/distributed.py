"""
DistributedDataParallel utilities.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_distributed():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_distributed():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(
    device_arg: str = "auto",
) -> tuple[torch.device, int, int, bool]:
    distributed = (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and "LOCAL_RANK" in os.environ
    )

    if not distributed:
        if device_arg == "auto":
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            device = torch.device(device_arg)

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device, 0, 0, False

    if not torch.cuda.is_available():
        raise RuntimeError(
            "DDP запущен, но CUDA недоступна."
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, "
            f"доступно GPU={torch.cuda.device_count()}."
        )

    torch.cuda.set_device(local_rank)

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

    return device, local_rank, rank, True


def cleanup_distributed() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def seed_everything(
    seed: int,
    rank: int = 0,
) -> None:
    final_seed = int(seed) + int(rank)

    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(final_seed)


def wrap_ddp(
    model: torch.nn.Module,
    device: torch.device,
    local_rank: int,
    distributed: bool,
) -> torch.nn.Module:
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
    if isinstance(model, DistributedDataParallel):
        return model.module

    if isinstance(model, torch.nn.DataParallel):
        return model.module

    return model


def reduce_mean(
    value: torch.Tensor,
) -> torch.Tensor:
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
    return reduce_mean(value)


def reduce_sum(
    value: torch.Tensor,
) -> torch.Tensor:
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
    if not is_distributed():
        return obj

    values = [obj]

    dist.broadcast_object_list(
        values,
        src=src,
    )

    return values[0]
