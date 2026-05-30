"""Tests for benchmark metrics."""

import pytest

from vlm_bench.benchmark.metrics import BenchmarkMetrics


class TestBenchmarkMetrics:
    def test_empty_metrics(self):
        m = BenchmarkMetrics()
        with pytest.raises(ValueError):
            m.compute_latency()

    def test_latency_computation(self):
        m = BenchmarkMetrics()
        for lat in [10.0, 20.0, 30.0, 40.0, 50.0]:
            m.record_latency(lat)
        result = m.compute_latency()
        assert result.mean_ms == 30.0
        assert result.median_ms == 30.0
        assert result.min_ms == 10.0
        assert result.max_ms == 50.0

    def test_reset(self):
        m = BenchmarkMetrics()
        m.record_latency(10.0)
        m.reset()
        with pytest.raises(ValueError):
            m.compute_latency()
