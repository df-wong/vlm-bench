"""GPU memory management utilities."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MemorySnapshot:
    """Point-in-time GPU memory snapshot."""
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    free_mb: float
    total_mb: float
    fragmentation_pct: float


class MemoryManager:
    """GPU memory management with tracking and cleanup."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self._snapshots: list[MemorySnapshot] = []

    def snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        if not torch.cuda.is_available():
            return MemorySnapshot(0, 0, 0, 0, 0, 0)

        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(self.device) / 1024**2
        reserved = torch.cuda.memory_reserved(self.device) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**2
        total = torch.cuda.get_device_properties(self.device).total_mem / 1024**2
        free = total - reserved

        snap = MemorySnapshot(
            allocated_mb=allocated,
            reserved_mb=reserved,
            max_allocated_mb=max_allocated,
            free_mb=free,
            total_mb=total,
            fragmentation_pct=((reserved - allocated) / reserved * 100) if reserved > 0 else 0,
        )
        self._snapshots.append(snap)
        return snap

    def cleanup(self) -> float:
        """Aggressively free GPU memory. Returns MB freed."""
        before = torch.cuda.memory_allocated(self.device) / 1024**2 if torch.cuda.is_available() else 0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        after = torch.cuda.memory_allocated(self.device) / 1024**2 if torch.cuda.is_available() else 0
        return before - after

    def get_optimal_batch_size(self, model_size_mb: float, safety_factor: float = 0.8) -> int:
        """Estimate optimal batch size given model size and available memory."""
        if not torch.cuda.is_available():
            return 1
        snap = self.snapshot()
        available = snap.free_mb * safety_factor
        per_sample_estimate = model_size_mb * 0.3  # rough estimate
        if per_sample_estimate <= 0:
            return 1
        return max(1, int(available / per_sample_estimate))
