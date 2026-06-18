#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G          # Increased memory for full parameter tuning
#SBATCH --time=10:00:00
#SBATCH --account=def-erangauk-ab

source # environment-path
export HF_HOME=$PWD/hf_cache
# Increase timeout to 2 hours (in seconds)
export NCCL_IB_TIMEOUT=22
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

accelerate launch --config_file accelerate_config.yaml train_vlm.py