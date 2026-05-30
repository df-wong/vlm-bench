"""Dataset loaders for VLM benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


class VLMDataset(Dataset):
    """Generic VLM dataset for benchmarking.

    Supports:
    - Image + question pairs from JSONL files
    - Image folders with auto-generated prompts
    - Synthetic data for throughput testing
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        transform=None,
        max_samples: Optional[int] = None,
        synthetic: bool = False,
        synthetic_size: tuple[int, int] = (336, 336),
    ):
        self.transform = transform
        self.samples = []

        if synthetic:
            self.samples = [
                {"image": None, "prompt": f"Describe image {i}", "id": i}
                for i in range(max_samples or 1000)
            ]
            self._synthetic_size = synthetic_size
        elif data_path:
            path = Path(data_path)
            if path.suffix == ".jsonl":
                with open(path) as f:
                    for i, line in enumerate(f):
                        if max_samples and i >= max_samples:
                            break
                        self.samples.append(json.loads(line))
            elif path.suffix == ".json":
                data = json.loads(path.read_text())
                self.samples = data[:max_samples] if max_samples else data

        self._image_dir = Path(image_dir) if image_dir else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        if sample.get("image") is None and not hasattr(self, "_synthetic_size"):
            image = Image.new("RGB", (336, 336), color=(128, 128, 128))
        elif hasattr(self, "_synthetic_size"):
            image = Image.new("RGB", self._synthetic_size, color=(128, 128, 128))
        else:
            img_path = sample["image"]
            if self._image_dir:
                img_path = self._image_dir / img_path
            image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "prompt": sample.get("prompt", sample.get("question", "")),
            "id": sample.get("id", idx),
        }
