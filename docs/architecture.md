# Architecture

## Overview

vlm-bench is designed as a modular framework for VLM inference and benchmarking.

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI (click)                          │
│  infer │ benchmark │ serve │ devices │ models               │
├─────────────────────────────────────────────────────────────┤
│                     Model Factory                           │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐                   │
│  │  LLaVA  │  │ InternVL │  │ Qwen-VL  │  ...              │
│  └────┬────┘  └────┬─────┘  └────┬─────┘                   │
│       └────────────┼─────────────┘                          │
│                    ▼                                         │
│  ┌─────────────────────────────────┐                        │
│  │         BaseVLM                 │                        │
│  │  load_model() │ generate()      │                        │
│  │  encode_image() │ unload()      │                        │
│  └────────────────┬────────────────┘                        │
│                   ▼                                          │
│  ┌─────────────────────────────────┐                        │
│  │         Backends                │                        │
│  │  PyTorch │ ROCm │ ONNX │ TRT   │                        │
│  └─────────────────────────────────┘                        │
├─────────────────────────────────────────────────────────────┤
│  Config (YAML + env vars) │ Benchmark Runner │ REST Server  │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

1. **Factory Pattern** — Models registered via decorators, instantiated by name
2. **Backend Abstraction** — Same model runs on PyTorch, ROCm, or ONNX without code changes
3. **Config Layering** — YAML base → environment variables → CLI overrides
4. **Dataclasses Everywhere** — Type-safe configuration and outputs
5. **Thread-safe Server** — FastAPI with async + executor for non-blocking inference

## Data Flow

```
Image + Prompt → VLMInput → Model.encode_image() → Model.generate() → VLMOutput
                                    ↓
                              Backend.prepare_model() → Backend.run_inference()
```

## Adding a New Model

1. Create `src/vlm_bench/models/your_model.py`
2. Implement `BaseVLM` subclass
3. Decorate with `@register_model("your-model-name")`
4. Import in `src/vlm_bench/models/__init__.py`
