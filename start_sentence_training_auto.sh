#!/bin/bash

echo "=================================="
echo "SENTENCE COMPLETION MODEL TRAINING"
echo "=================================="
echo ""
echo "Training with:"
echo "- High-quality sentence data"
echo "- Llama 3.2 3B model (already cached)"
echo "- Natural prompt-completion pairs"
echo ""

# Skip download since model is already cached
echo "Model already cached, skipping download..."
echo ""

# Run training directly without prompts
cd /Users/simonlacey/Documents/GitHub/llms/ai-train-llm

echo "Starting training..."
echo ""

python3 -m mlx_lm lora --config python/sentence_training/mlx/config_fixed.yaml

echo ""
echo "Training complete!"
echo ""
echo "Next steps:"
echo "1. Update mlx_model_server.py to use 'models/bio-sentence-llama3-lora'"
echo "2. Restart the MLX server"
echo "3. Test the improved completions"