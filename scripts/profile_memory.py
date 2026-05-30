#!/usr/bin/env python3
"""Profile GPU memory usage for different models and batch sizes."""

import argparse
import json

import torch

from vlm_bench.models.base import VLMInput
from vlm_bench.models.factory import ModelFactory
from vlm_bench.utils.memory import MemoryManager


def main():
    parser = argparse.ArgumentParser(description="Profile VLM memory usage")
    parser.add_argument("--model", "-m", required=True, help="Model name")
    parser.add_argument("--batch-sizes", "-b", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--output", "-o", default="memory_profile.json")
    args = parser.parse_args()

    mm = MemoryManager()
    results = []

    for bs in args.batch_sizes:
        print(f"\nBatch size: {bs}")

        # Clean start
        mm.cleanup()

        # Load model
        model = ModelFactory.create(args.model)
        model.load_model()
        after_load = mm.snapshot()
        print(f"  Model loaded: {after_load.allocated_mb:.0f} MB allocated")

        # Run inference
        dummy = VLMInput(prompt="Test", max_new_tokens=32)
        for _ in range(3):
            model.generate(dummy)

        after_infer = mm.snapshot()
        print(f"  After inference: {after_infer.allocated_mb:.0f} MB allocated, {after_infer.peak_mb if hasattr(after_infer, 'peak_mb') else after_infer.max_allocated_mb:.0f} MB peak")

        results.append({
            "batch_size": bs,
            "model_load_mb": after_load.allocated_mb,
            "inference_mb": after_infer.allocated_mb,
            "max_allocated_mb": after_infer.max_allocated_mb,
        })

        model.unload()

    Path = __import__("pathlib").Path
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
