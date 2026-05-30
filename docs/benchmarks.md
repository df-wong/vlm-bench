# Benchmark Methodology

## Metrics Collected

| Metric | Unit | Description |
|--------|------|-------------|
| Latency | ms | End-to-end inference time |
| Throughput | req/s | Requests processed per second |
| Tokens/s | tok/s | Generated tokens per second |
| Memory Peak | MB | Maximum GPU memory used |
| Memory Allocated | MB | Current GPU memory in use |

## Procedure

1. **Warmup** — 10 iterations (configurable) to stabilize GPU clocks and caches
2. **Measurement** — N iterations with CUDA/HIP synchronization
3. **Statistics** — Mean, median, p95, p99, min, max, std

## Running Benchmarks

```bash
# Single model
vlm-bench benchmark --model llava-v1.6 --iterations 100

# Multiple models
vlm-bench benchmark --model llava-v1.6 --model internvl-2 --iterations 50

# With config file
vlm-bench benchmark --config configs/benchmark.yaml

# Compare backends
vlm-bench benchmark --model llava-v1.6 --backend pytorch
vlm-bench benchmark --model llava-v1.6 --backend rocm
```

## Output Format

Results are saved as JSON:

```json
[
  {
    "model": "llava-v1.6",
    "backend": "pytorch",
    "batch_size": 1,
    "latency_ms": 145.3,
    "tokens_per_second": 34.2,
    "memory_peak_mb": 4521.0
  }
]
```
