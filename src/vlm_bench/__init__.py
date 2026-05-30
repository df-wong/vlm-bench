"""vlm-bench: Multi-modal VLM benchmarking & inference for AMD GPUs."""

__version__ = "0.2.0"
__author__ = "Difa Wong"

from vlm_bench.config import VLMConfig
from vlm_bench.models.factory import ModelFactory

__all__ = ["VLMConfig", "ModelFactory"]
