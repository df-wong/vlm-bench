"""Distributed inference utilities."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist


def setup_distributed(
    backend: str = "nccl",
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> tuple[int, int]:
    """Initialize distributed process group."""
    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())

    return rank, world_size


def cleanup_distributed() -> None:
    """Destroy distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is rank 0."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0
