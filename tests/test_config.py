"""Tests for configuration management."""

import os
import pytest
from pathlib import Path

from vlm_bench.config import VLMConfig, ModelConfig, BackendConfig, _deep_merge, _interpolate_env


class TestVLMConfig:
    def test_default_config(self):
        config = VLMConfig()
        assert config.model.name == "llava-v1.6"
        assert config.backend.name == "pytorch"
        assert config.seed == 42

    def test_from_yaml(self, tmp_config):
        config = VLMConfig.from_yaml(tmp_config)
        assert config.model.name == "llava-v1.6"
        assert config.backend.device == "cpu"
        assert config.benchmark.iterations == 5

    def test_from_yaml_not_found(self):
        with pytest.raises(FileNotFoundError):
            VLMConfig.from_yaml("/nonexistent/config.yaml")

    def test_env_interpolation(self, monkeypatch):
        monkeypatch.setenv("TEST_DEVICE", "cuda:1")
        result = _interpolate_env("${TEST_DEVICE}")
        assert result == "cuda:1"

    def test_env_interpolation_default(self):
        result = _interpolate_env("${NONEXISTENT_VAR:default_value}")
        assert result == "default_value"

    def test_to_yaml(self, tmp_path):
        config = VLMConfig()
        out_path = tmp_path / "out.yaml"
        config.to_yaml(out_path)
        assert out_path.exists()
        loaded = VLMConfig.from_yaml(out_path)
        assert loaded.model.name == config.model.name


class TestDeepMerge:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"model": {"name": "llava", "dtype": "fp16"}}
        override = {"model": {"dtype": "bf16"}}
        result = _deep_merge(base, override)
        assert result["model"]["name"] == "llava"
        assert result["model"]["dtype"] == "bf16"
