#!/bin/bash
# Start training the sentence completion model

echo "=================================="
echo "SENTENCE COMPLETION MODEL TRAINING"
echo "=================================="
echo ""
echo "This will train a new model using:"
echo "- High-quality sentence data (no repetitive patterns)"
echo "- Better base model than Phi-3"
echo "- Natural prompt-completion pairs"
echo ""

# Navigate to project directory
cd "$(dirname "$0")"

# First, download the base model if needed
echo "Step 1: Downloading base model..."
echo "=================================="
python3 python/mlx_training/download_base_model.py

# Check if download was successful
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Model download failed. Please fix the issue and try again."
    exit 1
fi

echo ""
echo "Step 2: Starting training..."
echo "=================================="

# Run the training script
python3 python/mlx_training/train_sentence_model.py

echo ""
echo "Training pipeline completed."