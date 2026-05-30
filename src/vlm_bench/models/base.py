"""Abstract base class for Vision-Language Models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch
from PIL import Image


@dataclass
class VLMOutput:
    """Standardized output from VLM inference."""
    text: str
    tokens_generated: int
    latency_ms: float
    tokens_per_second: float
    memory_used_mb: Optional[float] = None
    logits: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None


@dataclass
class VLMInput:
    """Standardized input for VLM inference."""
    image: Optional[Image.Image] = None
    image_path: Optional[str] = None
    prompt: str = ""
    system_prompt: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.0


class BaseVLM(ABC):
    """Abstract base class for all VLM implementations.

    Subclasses must implement:
    - load_model(): Load weights and prepare for inference
    - generate(): Run text generation given image + prompt
    - encode_image(): Process image into model-compatible format
    """

    def __init__(self, model_name: str, device: str = "cuda:0", dtype: str = "float16"):
        self.model_name = model_name
        self.device = device
        self.dtype = getattr(torch, dtype)
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load_model(self, cache_dir: Optional[str] = None) -> None:
        """Load model weights and processors."""
        ...

    @abstractmethod
    def generate(self, vlm_input: VLMInput) -> VLMOutput:
        """Run inference: image + prompt -> text output."""
        ...

    @abstractmethod
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Process raw PIL image into model-ready tensor."""
        ...

    def unload(self) -> None:
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_memory_usage(self) -> dict[str, float]:
        """Get current GPU memory usage in MB."""
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**2,
            "reserved": torch.cuda.memory_reserved() / 1024**2,
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**2,
        }

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"{self.__class__.__name__}(name={self.model_name}, device={self.device}, {status})"
