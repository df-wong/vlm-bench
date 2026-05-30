"""Model factory — registry pattern for VLM instantiation."""

from __future__ import annotations

from typing import Type

from vlm_bench.models.base import BaseVLM

# Global registry
_REGISTRY: dict[str, Type[BaseVLM]] = {}


def register_model(name: str):
    """Decorator to register a VLM class in the factory."""
    def decorator(cls: Type[BaseVLM]) -> Type[BaseVLM]:
        _REGISTRY[name.lower()] = cls
        return cls
    return decorator


class ModelFactory:
    """Factory for creating VLM instances by name."""

    @staticmethod
    def create(
        model_name: str,
        device: str = "cuda:0",
        dtype: str = "float16",
        **kwargs,
    ) -> BaseVLM:
        """Create a VLM instance by registered name.

        Args:
            model_name: Registered model name (e.g., "llava-v1.6", "internvl-2")
            device: Target device
            dtype: Model dtype
            **kwargs: Additional model-specific arguments

        Returns:
            Initialized (but not loaded) VLM instance

        Raises:
            ValueError: If model_name not registered
        """
        name_lower = model_name.lower()

        # Try exact match first
        if name_lower in _REGISTRY:
            return _REGISTRY[name_lower](model_name, device, dtype, **kwargs)

        # Try partial match
        for registered_name, cls in _REGISTRY.items():
            if registered_name in name_lower or name_lower in registered_name:
                return cls(model_name, device, dtype, **kwargs)

        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model: {model_name}. Available: {available}"
        )

    @staticmethod
    def list_models() -> list[str]:
        """List all registered model names."""
        return sorted(_REGISTRY.keys())

    @staticmethod
    def register(name: str, cls: Type[BaseVLM]) -> None:
        """Manually register a model class."""
        _REGISTRY[name.lower()] = cls
