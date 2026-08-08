FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Remove stale Yarn repo (expired GPG key in the base image breaks apt-get update)
RUN rm -f /etc/apt/sources.list.d/yarn.list

# Only the tools needed to orchestrate cluster jobs from the laptop:
#   openssh-client -> SSH tunnels to the Empire AI login node (see CLAUDE.md)
#   git/curl/wget  -> repo + downloads
# No build-essential / CUDA / ML stack: real compute runs on the cluster, not here.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*
