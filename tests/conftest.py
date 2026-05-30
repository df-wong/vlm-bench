"""Pytest fixtures for vlm-bench tests."""

import pytest
from pathlib import Path


@pytest.fixture
def tmp_config(tmp_path):
    """Create a temporary config YAML file."""
    config_content = """
model:
  name: llava-v1.6
  variant: 7b
  dtype: float16

backend:
  name: pytorch
  device: cpu

benchmark:
  warmup_iterations: 2
  iterations: 5
  batch_sizes: [1]
  output_dir: /tmp/vlm-bench-test

log_level: DEBUG
seed: 42
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample test image."""
    from PIL import Image
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    img_path = tmp_path / "test.jpg"
    img.save(img_path)
    return img_path
