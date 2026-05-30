# ROCm Setup Guide

## Prerequisites

- AMD GPU (MI200/MI300 series, RX 7000 series)
- Ubuntu 22.04/24.04 LTS
- ROCm 6.0+ installed

## Installation

### 1. Install ROCm

```bash
# Add ROCm repository
sudo apt update && sudo apt install -y wget gnupg
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.0 jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update
sudo apt install -y rocm-hip-sdk rocm-hip-runtime rocm-dev

# Add user to render group
sudo usermod -aG render,video $USER
```

### 2. Verify Installation

```bash
# Check GPU
rocm-smi

# Check HIP
hipcc --version

# Python check
python -c "import torch; print(torch.version.hip); print(torch.cuda.get_device_name(0))"
```

### 3. Install vlm-bench with ROCm

```bash
pip install -e ".[rocm,server]"
```

### 4. FlashAttention-ROCm

```bash
# Install FlashAttention for ROCm
pip install flash-attn --no-build-isolation
```

## Environment Variables

```bash
# Memory allocation (recommended)
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"

# Visible devices
export HIP_VISIBLE_DEVICES=0

# Disable CUBLAS workspaces (save memory)
export CUBLAS_WORKSPACE_CONFIG=:16:8
```

## Troubleshooting

### GPU not detected
```bash
# Check if GPU is visible
rocm-smi
ls /dev/kfd /dev/dri/render*

# Fix permissions
sudo chmod 666 /dev/kfd /dev/dri/*
```

### Out of memory
```bash
# Enable expandable segments
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"

# Use gradient checkpointing
# In config: model.gradient_checkpointing=true
```
