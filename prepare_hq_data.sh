#!/bin/bash

# Prepare high-quality training data with grammar filtering

echo "Preparing high-quality training data with grammar filtering..."

# Change to MLX training directory
cd python/mlx_training

# Activate virtual environment
source ../venv/bin/activate

# Install required dependencies
echo "Installing dependencies..."
pip install spacy language-tool-python

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the improved data preparation
echo "Processing LookingFor dataset with quality filters..."
python prepare_lookingfor_mlx_improved.py

echo "Data preparation complete!"
echo "High-quality dataset saved to: python/mlx_training/lookingfor_hq/"

# Show statistics
echo -e "\nDataset statistics:"
echo "Train samples: $(wc -l < lookingfor_hq/train.jsonl)"
echo "Valid samples: $(wc -l < lookingfor_hq/valid.jsonl)"
echo "Test samples: $(wc -l < lookingfor_hq/test.jsonl)"