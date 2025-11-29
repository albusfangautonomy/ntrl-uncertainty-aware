# ARM64 Python 3.10 base image
FROM --platform=linux/arm64 python:3.10-slim-bullseye

# Install system dependencies for scipy, matplotlib, libigl, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install latest stable CPU-only PyTorch for ARM
RUN pip install --no-cache-dir torch torchvision torchaudio

# Install additional Python deps
RUN pip install --no-cache-dir \
    libigl \
    pytorch_kinematics \
    configargparse \
    scipy \
    matplotlib

WORKDIR /workspace
