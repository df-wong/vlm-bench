.PHONY: install test lint typecheck benchmark clean docker

install:
	pip install -e ".[dev,rocm,server,benchmark]"

test:
	pytest tests/ -v --cov=vlm_bench --cov-report=term-missing

lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/vlm_bench/

benchmark:
	vlm-bench benchmark --config configs/benchmark.yaml --output results/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

docker:
	docker build -t vlm-bench:latest -f docker/Dockerfile.rocm .

docker-run:
	docker compose up vlm-bench

profile:
	python scripts/profile_memory.py --model llava-v1.6 --backend rocm
