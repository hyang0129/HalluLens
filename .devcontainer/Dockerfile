FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Remove stale Yarn repo (expired GPG key in base image)
RUN rm -f /etc/apt/sources.list.d/yarn.list

# Install system dependencies for vLLM, LMDB, and scientific computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    libffi-dev \
    libssl-dev \
    liblmdb-dev \
    && rm -rf /var/lib/apt/lists/*

# Pre-install PyTorch with CUDA support (speeds up postCreateCommand)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
