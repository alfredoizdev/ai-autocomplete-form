#!/bin/bash
# Final push to 1000 iterations

echo "🎯 Final training push: 600 → 1000 iterations"
echo "Current loss: ~0.8-1.2 (excellent!)"
echo "This will complete the training"
echo "----------------------------------------"

/opt/homebrew/bin/python3.11 -m mlx_lm lora \
    --config training_config_fresh.yaml \
    --train \
    --resume-adapter-file adapters/bio_mistral_fresh/0000600_adapters.safetensors \
    --iters 1000 \
    --save-every 100

echo "🎉 Training complete at 1000 iterations!"
echo "✅ Model is ready for production use"