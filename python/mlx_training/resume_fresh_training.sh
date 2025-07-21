#!/bin/bash
# Resume training from the fresh model checkpoint

echo "🚀 Resuming training from iteration 500..."
echo "Target: 2000 total iterations"
echo "----------------------------------------"

/opt/homebrew/bin/python3.11 -m mlx_lm lora \
    --config training_config_fresh.yaml \
    --train \
    --resume-adapter-file adapters/bio_mistral_fresh/0000500_adapters.safetensors \
    --iters 2000

echo "✅ Training complete!"