#!/usr/bin/env python3
"""Run comprehensive benchmarks and generate comparison tables."""

import argparse
import json
from pathlib import Path

from vlm_bench.benchmark.runner import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="Run VLM benchmarks")
    parser.add_argument("--models", "-m", nargs="+", default=["llava-v1.6"])
    parser.add_argument("--backends", "-b", nargs="+", default=["pytorch", "rocm"])
    parser.add_argument("--iterations", "-n", type=int, default=100)
    parser.add_argument("--output", "-o", default="results/")
    args = parser.parse_args()

    all_results = []

    for backend in args.backends:
        runner = BenchmarkRunner(
            models=args.models,
            backend=backend,
            iterations=args.iterations,
            output_dir=f"{args.output}/{backend}/",
        )
        results = runner.run()
        all_results.extend(results)

    # Save combined results
    output_path = Path(args.output) / "combined_results.json"
    output_path.write_text(json.dumps([r.__dict__ for r in all_results], indent=2, default=str))
    print(f"\nCombined results saved to {output_path}")


if __name__ == "__main__":
    main()
