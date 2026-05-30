"""CLI entry point — click-based command interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(package_name="vlm-bench")
def main():
    """vlm-bench: VLM benchmarking & inference for AMD GPUs."""
    pass


@main.command()
@click.option("--model", "-m", required=True, help="Model name (e.g., llava-v1.6, internvl-2)")
@click.option("--image", "-i", type=click.Path(exists=True), help="Image file path")
@click.option("--prompt", "-p", required=True, help="Text prompt")
@click.option("--device", "-d", default="cuda:0", help="Target device")
@click.option("--dtype", default="float16", help="Model dtype")
@click.option("--max-tokens", default=512, type=int, help="Max new tokens")
@click.option("--temperature", default=0.7, type=float, help="Sampling temperature")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file")
def infer(model, image, prompt, device, dtype, max_tokens, temperature, output):
    """Run VLM inference on a single image + prompt."""
    from vlm_bench.config import VLMConfig
    from vlm_bench.models.base import VLMInput
    from vlm_bench.models.factory import ModelFactory

    console.print(f"[bold blue]Loading model:[/] {model}")
    vlm = ModelFactory.create(model, device=device, dtype=dtype)
    vlm.load_model()

    vlm_input = VLMInput(
        image_path=image,
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )

    console.print("[bold green]Running inference...[/]")
    result = vlm.generate(vlm_input)

    console.print(f"\n[bold]Response:[/] {result.text}")
    console.print(f"[dim]Tokens: {result.tokens_generated} | Latency: {result.latency_ms:.1f}ms | Speed: {result.tokens_per_second:.1f} tok/s[/]")

    if result.memory_used_mb:
        console.print(f"[dim]Memory: {result.memory_used_mb:.1f} MB[/]")

    if output:
        out = {
            "text": result.text,
            "tokens": result.tokens_generated,
            "latency_ms": result.latency_ms,
            "tokens_per_second": result.tokens_per_second,
        }
        Path(output).write_text(json.dumps(out, indent=2))
        console.print(f"[dim]Saved to {output}[/]")


@main.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Benchmark config YAML")
@click.option("--model", "-m", multiple=True, help="Model(s) to benchmark")
@click.option("--backend", "-b", default="pytorch", help="Backend (pytorch, rocm)")
@click.option("--batch-size", default=1, type=int, help="Batch size")
@click.option("--iterations", "-n", default=100, type=int, help="Iterations")
@click.option("--output", "-o", default="results/", type=click.Path(), help="Output directory")
@click.option("--profile", is_flag=True, help="Enable profiling")
def benchmark(config, model, backend, batch_size, iterations, output, profile):
    """Run standardized VLM benchmarks."""
    from vlm_bench.benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(
        config_path=config,
        models=list(model) if model else None,
        backend=backend,
        batch_size=batch_size,
        iterations=iterations,
        output_dir=output,
        profile=profile,
    )
    runner.run()


@main.command()
@click.option("--model", "-m", required=True, help="Model name")
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=8000, type=int, help="Server port")
@click.option("--workers", default=1, type=int, help="Number of workers")
@click.option("--max-batch", default=8, type=int, help="Max batch size")
def serve(model, host, port, workers, max_batch):
    """Start VLM inference server."""
    from vlm_bench.server.app import create_app

    app = create_app(model_name=model, max_batch_size=max_batch)
    console.print(f"[bold blue]Starting server on {host}:{port}[/]")
    import uvicorn
    uvicorn.run(app, host=host, port=port, workers=workers)


@main.command()
def devices():
    """List available compute devices."""
    import torch

    table = Table(title="Compute Devices")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Memory", style="yellow")
    table.add_column("Backend", style="magenta")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            name = props.name
            mem = f"{props.total_mem / 1024**3:.1f} GB"
            backend = "ROCm" if ("AMD" in name or "MI" in name) else "CUDA"
            table.add_row(str(i), name, mem, backend)
    else:
        table.add_row("cpu", "CPU", "N/A", "PyTorch")

    console.print(table)


@main.command()
def models():
    """List supported models."""
    from vlm_bench.models.factory import ModelFactory

    table = Table(title="Supported Models")
    table.add_column("Name", style="cyan")

    for name in ModelFactory.list_models():
        table.add_row(name)

    console.print(table)


if __name__ == "__main__":
    main()
