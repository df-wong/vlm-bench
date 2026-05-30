# ROCm-enabled Docker image for vlm-bench
FROM rocm/pytorch:rocm6.0_ubuntu22.04_py3.10_pytorch_2.1.2

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy and install
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN pip install --no-cache-dir -e ".[rocm,server,benchmark]"

# Copy configs
COPY configs/ configs/

EXPOSE 8000

# Set ROCm env
ENV PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
ENV HIP_VISIBLE_DEVICES=0

ENTRYPOINT ["vlm-bench"]
CMD ["serve", "--model", "llava-v1.6", "--port", "8000"]
