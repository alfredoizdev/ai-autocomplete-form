#!/bin/bash

# MLX training using community model that doesn't require authentication

echo "Starting improved MLX training for Llama-3.2-3B (MLX Community)..."

# Change to MLX training directory
cd python/mlx_training

# Ensure virtual environment is activated
source ../venv/bin/activate

# Model configuration - using MLX community model
MODEL="mlx-community/Llama-3.2-3B-Instruct-4bit"
DATA_DIR="lookingfor_hq"  # High-quality dataset

# Training parameters optimized for quality
python -m mlx_lm lora \
    --model $MODEL \
    --train \
    --data $DATA_DIR \
    --batch-size 4 \
    --learning-rate 1e-5 \
    --iters 2000 \
    --val-batches 50 \
    --save-every 200 \
    --adapter-path "adapters/llama3.2-3b-lookingfor-hq" \
    --num-layers 24 \
    --steps-per-report 10 \
    --steps-per-eval 100

echo "Training complete! Adapter saved to adapters/llama3.2-3b-lookingfor-hq"