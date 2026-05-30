#!/usr/bin/env python3
"""Download and cache VLM models."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = {
    "llava-v1.6-7b": "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-v1.6-13b": "llava-hf/llava-v1.6-vicuna-13b-hf",
    "internvl-2-8b": "OpenGVLab/InternVL2-8B",
}


def main():
    parser = argparse.ArgumentParser(description="Download VLM models")
    parser.add_argument("--model", "-m", choices=list(MODELS.keys()), help="Specific model to download")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--cache-dir", default=None, help="Cache directory")
    args = parser.parse_args()

    models = MODELS if args.all else {args.model: MODELS[args.model]}

    for name, repo_id in models.items():
        print(f"Downloading {name} ({repo_id})...")
        snapshot_download(
            repo_id=repo_id,
            cache_dir=args.cache_dir,
            ignore_patterns=["*.bin", "*.safetensors.index.json"],
        )
        print(f"  ✓ {name} downloaded")


if __name__ == "__main__":
    main()
