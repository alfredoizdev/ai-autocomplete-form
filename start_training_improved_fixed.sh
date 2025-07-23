#!/bin/bash

# Improved MLX training script with supported parameters

echo "Starting improved MLX training for Llama-3.2-3B..."

# Change to MLX training directory
cd python/mlx_training

# Ensure virtual environment is activated
source ../venv/bin/activate

# Model configuration
MODEL="meta-llama/Llama-3.2-3B-Instruct"
DATA_DIR="lookingfor_hq"  # High-quality dataset

# Training parameters optimized for quality (using only supported args)
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

# Key improvements:
# 1. Lower learning rate (1e-5 vs 5e-5) for stability
# 2. Larger batch size (4) for better gradients
# 3. More layers (24 vs 16) for better representation
# 4. More iterations (2000) for thorough training
# 5. Frequent reporting and evaluation