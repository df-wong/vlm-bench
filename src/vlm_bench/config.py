"""Configuration management with YAML overrides and environment variable interpolation."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _interpolate_env(value: str) -> str:
    """Replace ${VAR} patterns with environment variable values."""
    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        default = None
        if ":" in var_name:
            var_name, default = var_name.split(":", 1)
        env_val = os.environ.get(var_name)
        if env_val is not None:
            return env_val
        if default is not None:
            return default
        raise ValueError(f"Environment variable {var_name} not set and no default provided")
    return _ENV_PATTERN.sub(_replace, value)


def _process_config(obj: Any) -> Any:
    """Recursively interpolate environment variables in config."""
    if isinstance(obj, str):
        return _interpolate_env(obj)
    elif isinstance(obj, dict):
        return {k: _process_config(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_process_config(item) for item in obj]
    return obj


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "llava-v1.6"
    variant: str = "7b"
    dtype: str = "float16"
    max_length: int = 2048
    trust_remote_code: bool = False
    cache_dir: Optional[str] = None
    attn_implementation: Optional[str] = None  # "flash_attention_2", "sdpa", "eager"


@dataclass
class BackendConfig:
    """Backend configuration."""
    name: str = "pytorch"  # pytorch, onnx, rocm, tensorrt
    device: str = "cuda:0"
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"
    quantize: Optional[str] = None  # "int8", "int4", "fp8"
    kv_cache_dtype: Optional[str] = None
    use_cuda_graph: bool = False


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    warmup_iterations: int = 10
    iterations: int = 100
    batch_sizes: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    input_sizes: list[tuple[int, int]] = field(default_factory=lambda: [(224, 224), (336, 336)])
    sequence_lengths: list[int] = field(default_factory=lambda: [128, 512, 1024])
    measure_memory: bool = true
    profile: bool = false
    output_dir: str = "results/"


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_batch_size: int = 8
    max_queue_size: int = 100
    timeout: float = 60.0
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class VLMConfig:
    """Root configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    log_level: str = "INFO"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: Optional[dict] = None) -> "VLMConfig":
        """Load config from YAML file with optional overrides and env interpolation."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        raw = _process_config(raw)

        if overrides:
            raw = _deep_merge(raw, overrides)

        return cls(
            model=ModelConfig(**raw.get("model", {})),
            backend=BackendConfig(**raw.get("backend", {})),
            benchmark=BenchmarkConfig(**raw.get("benchmark", {})),
            server=ServerConfig(**raw.get("server", {})),
            log_level=raw.get("log_level", "INFO"),
            seed=raw.get("seed", 42),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save config to YAML file."""
        import dataclasses
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts, override takes precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
