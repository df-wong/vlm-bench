"""Benchmark runner — orchestrates model benchmarks across configurations."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch

from vlm_bench.benchmark.metrics import BenchmarkMetrics
from vlm_bench.config import VLMConfig


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""
    model: str
    backend: str
    batch_size: int
    input_size: tuple[int, int]
    sequence_length: int
    latency_ms: float
    throughput: float
    tokens_per_second: float
    memory_peak_mb: float
    memory_allocated_mb: float
    iterations: int
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


class BenchmarkRunner:
    """Orchestrates VLM benchmarks with configurable parameters.

    Runs models through a matrix of batch sizes, input sizes, and sequence
    lengths, collecting latency, throughput, and memory metrics.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        models: Optional[list[str]] = None,
        backend: str = "pytorch",
        batch_size: int = 1,
        iterations: int = 100,
        output_dir: str = "results/",
        profile: bool = False,
    ):
        if config_path:
            self.config = VLMConfig.from_yaml(config_path)
        else:
            self.config = VLMConfig()

        self.models = models or ["llava-v1.6"]
        self.backend = backend
        self.batch_size = batch_size
        self.iterations = iterations
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.results: list[BenchmarkResult] = []

    def run(self) -> list[BenchmarkResult]:
        """Execute full benchmark suite."""
        from vlm_bench.backends.rocm_backend import ROCmBackend
        from vlm_bench.models.factory import ModelFactory

        console_print = lambda msg: print(f"[bench] {msg}")

        for model_name in self.models:
            console_print(f"Benchmarking {model_name} with {self.backend} backend...")

            # Create model
            vlm = ModelFactory.create(model_name, device="cuda:0")
            vlm.load_model()

            # Create backend
            if self.backend == "rocm":
                be = ROCmBackend(device="cuda:0")
            else:
                from vlm_bench.backends.pytorch_backend import PyTorchBackend
                be = PyTorchBackend(device="cuda:0")

            # Run benchmarks
            for bs in [self.batch_size]:
                result = self._benchmark_single(vlm, be, model_name, bs)
                self.results.append(result)
                console_print(
                    f"  batch={bs}: {result.latency_ms:.1f}ms, "
                    f"{result.tokens_per_second:.1f} tok/s, "
                    f"{result.memory_peak_mb:.0f} MB"
                )

            vlm.unload()

        self._save_results()
        self._print_summary()
        return self.results

    def _benchmark_single(self, vlm, backend, model_name: str, batch_size: int) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        from vlm_bench.models.base import VLMInput

        # Warmup
        dummy_input = VLMInput(prompt="Describe this image.", max_new_tokens=32)
        for _ in range(5):
            vlm.generate(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measure
        latencies = []
        tokens_list = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            result = vlm.generate(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
            tokens_list.append(result.tokens_generated)

        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = sum(tokens_list) / len(tokens_list)

        return BenchmarkResult(
            model=model_name,
            backend=backend,
            batch_size=batch_size,
            input_size=(336, 336),
            sequence_length=512,
            latency_ms=avg_latency,
            throughput=batch_size / (avg_latency / 1000),
            tokens_per_second=avg_tokens / (avg_latency / 1000),
            memory_peak_mb=torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            memory_allocated_mb=torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            iterations=self.iterations,
        )

    def _save_results(self) -> None:
        """Save results to JSON."""
        output_file = self.output_dir / "benchmark_results.json"
        data = [asdict(r) for r in self.results]
        output_file.write_text(json.dumps(data, indent=2))
        print(f"[bench] Results saved to {output_file}")

    def _print_summary(self) -> None:
        """Print benchmark summary table."""
        print("\n" + "=" * 80)
        print(f"{'Model':<20} {'Backend':<10} {'Latency':<12} {'Tok/s':<10} {'Memory':<12}")
        print("-" * 80)
        for r in self.results:
            print(f"{r.model:<20} {r.backend:<10} {r.latency_ms:<12.1f} {r.tokens_per_second:<10.1f} {r.memory_peak_mb:<12.0f}")
        print("=" * 80)
