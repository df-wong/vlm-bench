"""Benchmark metrics collection and analysis."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LatencyMetrics:
    """Latency statistics."""
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float


@dataclass
class MemoryMetrics:
    """GPU memory statistics."""
    peak_mb: float
    allocated_mb: float
    reserved_mb: float
    fragmentation_ratio: float


@dataclass
class ThroughputMetrics:
    """Throughput statistics."""
    tokens_per_second: float
    requests_per_second: float
    batch_efficiency: float


class BenchmarkMetrics:
    """Collects and analyzes benchmark measurements."""

    def __init__(self):
        self._latencies: list[float] = []
        self._memory_snapshots: list[dict[str, float]] = []
        self._token_counts: list[int] = []

    def record_latency(self, latency_ms: float) -> None:
        self._latencies.append(latency_ms)

    def record_memory(self) -> None:
        if torch.cuda.is_available():
            self._memory_snapshots.append({
                "allocated": torch.cuda.memory_allocated() / 1024**2,
                "reserved": torch.cuda.memory_reserved() / 1024**2,
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**2,
            })

    def record_tokens(self, count: int) -> None:
        self._token_counts.append(count)

    def compute_latency(self) -> LatencyMetrics:
        if not self._latencies:
            raise ValueError("No latency measurements recorded")
        sorted_lat = sorted(self._latencies)
        n = len(sorted_lat)
        return LatencyMetrics(
            mean_ms=statistics.mean(sorted_lat),
            median_ms=statistics.median(sorted_lat),
            p95_ms=sorted_lat[int(n * 0.95)],
            p99_ms=sorted_lat[int(n * 0.99)],
            min_ms=sorted_lat[0],
            max_ms=sorted_lat[-1],
            std_ms=statistics.stdev(sorted_lat) if n > 1 else 0.0,
        )

    def compute_memory(self) -> MemoryMetrics:
        if not self._memory_snapshots:
            return MemoryMetrics(0, 0, 0, 0)
        peak = max(s["max_allocated"] for s in self._memory_snapshots)
        alloc = statistics.mean(s["allocated"] for s in self._memory_snapshots)
        reserved = statistics.mean(s["reserved"] for s in self._memory_snapshots)
        return MemoryMetrics(
            peak_mb=peak,
            allocated_mb=alloc,
            reserved_mb=reserved,
            fragmentation_ratio=(reserved - alloc) / reserved if reserved > 0 else 0,
        )

    def compute_throughput(self, batch_size: int = 1) -> ThroughputMetrics:
        if not self._latencies:
            raise ValueError("No measurements recorded")
        avg_latency_s = statistics.mean(self._latencies) / 1000
        total_tokens = sum(self._token_counts) if self._token_counts else 0
        avg_tokens = total_tokens / len(self._latencies) if self._latencies else 0
        return ThroughputMetrics(
            tokens_per_second=avg_tokens / avg_latency_s if avg_latency_s > 0 else 0,
            requests_per_second=1.0 / avg_latency_s if avg_latency_s > 0 else 0,
            batch_efficiency=batch_size / (avg_latency_s * 100) if avg_latency_s > 0 else 0,
        )

    def reset(self) -> None:
        self._latencies.clear()
        self._memory_snapshots.clear()
        self._token_counts.clear()
