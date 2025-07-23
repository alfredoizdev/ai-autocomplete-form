#!/bin/bash

# Fast high-quality data preparation

echo "Preparing high-quality training data (fast version)..."

# Change to MLX training directory
cd python/mlx_training

# Activate virtual environment
source ../venv/bin/activate

# Run the fast data preparation
echo "Processing LookingFor dataset with quality filters..."
python prepare_lookingfor_mlx_fast.py

echo "Data preparation complete!"
echo "High-quality dataset saved to: python/mlx_training/lookingfor_hq/"

# Show statistics
echo -e "\nDataset statistics:"
echo "Train samples: $(wc -l < lookingfor_hq/train.jsonl)"
echo "Valid samples: $(wc -l < lookingfor_hq/valid.jsonl)"
echo "Test samples: $(wc -l < lookingfor_hq/test.jsonl)"