#!/bin/bash
# Final training push to iteration 1000

echo "🚀 Final training push to iteration 1000"
echo "Current model performs well, this will polish it further"
echo "----------------------------------------"

/opt/homebrew/bin/python3.11 -m mlx_lm lora \
    --config training_config_fresh.yaml \
    --train \
    --resume-adapter-file adapters/bio_mistral_fresh/adapters.safetensors \
    --iters 1000 \
    --save-every 100

echo "✅ Training to iteration 1000 complete!"