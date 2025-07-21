#!/bin/bash
# Continue training from 600 to 1000 iterations

echo "🚀 Continuing training from 600 to 1000 iterations"
echo "Current loss: ~0.9-1.2 (excellent!)"
echo "----------------------------------------"

/opt/homebrew/bin/python3.11 -m mlx_lm lora \
    --config training_config_fresh.yaml \
    --train \
    --resume-adapter-file adapters/bio_mistral_fresh/0000600_adapters.safetensors \
    --iters 1000 \
    --save-every 100

echo "✅ Training complete at 1000 iterations!"