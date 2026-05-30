# 🔬 vlm-bench

**Multi-modal Vision-Language Model Benchmarking & Inference Framework**

Optimized for AMD GPUs with ROCm/HIP acceleration. Supports LLaVA, InternVL, Qwen-VL, and custom architectures.

[![CI](https://github.com/df-wong/vlm-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/df-wong/vlm-bench/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Features

- 🚀 **Multi-backend inference** — PyTorch, ONNX Runtime, ROCm/HIP, TensorRT
- 📊 **Comprehensive benchmarking** — Latency, throughput, memory, token/s metrics
- 🔌 **REST API server** — FastAPI-based with streaming, batching, health checks
- 🎯 **Model zoo** — LLaVA-1.6, InternVL-2, Qwen-VL-Chat, custom VLM support
- 🖥️ **AMD GPU optimized** — ROCm 6.x, HIP kernels, FlashAttention-ROCm
- 📦 **Docker ready** — ROCm-enabled containers with compose profiles
- 🧪 **Tested** — pytest + benchmark regression tests

## Quick Start

```bash
# Install
pip install -e ".[rocm,server]"

# Run inference
vlm-bench infer --model llava-v1.6 --image test.jpg --prompt "Describe this image"

# Benchmark
vlm-bench benchmark --model llava-v1.6 --backend rocm --batch-size 4 --iterations 100

# Start server
vlm-bench serve --model internvl-2 --port 8000 --workers 4
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   CLI/API   │────▶│  Model Zoo   │────▶│  Backends   │
│  (FastAPI)  │     │  (Factory)   │     │ PyTorch/ROCm│
└─────────────┘     └──────────────┘     │ ONNX/TRT    │
       │                   │              └─────────────┘
       ▼                   ▼                    │
┌─────────────┐     ┌──────────────┐           ▼
│  Benchmark  │     │   Dataset    │     ┌─────────────┐
│   Runner    │     │   Loaders    │     │  AMD GPU    │
│  (Metrics)  │     │ (Transforms) │     │  Hardware   │
└─────────────┘     └──────────────┘     └─────────────┘
```

## Supported Models

| Model | Params | Task | Backend |
|-------|--------|------|---------|
| LLaVA-v1.6 | 7B/13B | VQA, Caption | PyTorch, ROCm |
| InternVL-2 | 8B/26B | VQA, OCR, Doc | PyTorch, ROCm |
| Qwen-VL-Chat | 7B | VQA, Grounding | PyTorch, ROCm |

## ROCm Setup

See [docs/rocm_setup.md](docs/rocm_setup.md) for AMD GPU configuration.

```bash
# Check ROCm installation
rocm-smi

# Verify HIP
hipcc --version

# Install with ROCm support
pip install -e ".[rocm]"
```

## Benchmarks

Run standardized benchmarks across backends:

```bash
vlm-bench benchmark --config configs/benchmark.yaml --output results/
```

See [docs/benchmarks.md](docs/benchmarks.md) for methodology and result interpretation.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev,rocm,server]"

# Run tests
make test

# Lint
make lint

# Type check
make typecheck
```

## Citation

```bibtex
@software{vlm-bench2026,
  author = {Difa Wong},
  title = {vlm-bench: VLM Benchmarking for AMD GPUs},
  year = {2026},
  url = {https://github.com/df-wong/vlm-bench}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
