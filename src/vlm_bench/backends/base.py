"""Abstract base for inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class BackendInfo:
    """Backend metadata."""
    name: str
    version: str
    device: str
    supports_quantization: bool = False
    supports_cuda_graph: bool = False
    supports_flash_attention: bool = False


class BaseBackend(ABC):
    """Abstract inference backend."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self._compiled = False

    @abstractmethod
    def info(self) -> BackendInfo:
        """Return backend capabilities."""
        ...

    @abstractmethod
    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize/compile model for this backend."""
        ...

    @abstractmethod
    def run_inference(self, model: torch.nn.Module, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute forward pass."""
        ...

    def warmup(self, model: torch.nn.Module, inputs: dict[str, Any], iterations: int = 10) -> None:
        """Warmup the backend with dummy inference runs."""
        for _ in range(iterations):
            self.run_inference(model, inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
