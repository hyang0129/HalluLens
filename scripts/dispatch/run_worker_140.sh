#!/bin/bash
cd /mnt/home/hyang1/LLM_research/HalluLens
export PROJECT_ROOT=$PWD
export DISPATCH_ROOT=$PWD/shared/issue_140_dispatch
export DISPATCH_NODE=$(hostname)-140
exec bash scripts/dispatch/worker_79.sh
