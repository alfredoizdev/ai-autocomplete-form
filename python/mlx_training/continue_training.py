#!/usr/bin/env python3
"""Continue training from the last checkpoint."""

import subprocess
import sys
import os

# Configuration
config_file = "training_config.yaml"
adapter_checkpoint = "adapters/bio_mistral_lora/0000500_adapters.safetensors"

# Update the config to train for more iterations
print("Continuing training from iteration 500...")
print("This will train for additional iterations to improve the model.")
print("-" * 50)

# Build the training command
cmd = [
    sys.executable,
    "-m", "mlx_lm", "lora",
    "--config", config_file,
    "--train",
    "--resume-adapter-file", adapter_checkpoint,
    "--iters", "2000",  # Total iterations (will continue from 500)
]

print(f"Command: {' '.join(cmd)}")
print("-" * 50)

# Execute training
try:
    subprocess.run(cmd, check=True)
    print("\n✅ Training completed successfully!")
except subprocess.CalledProcessError as e:
    print(f"\n❌ Training failed with error: {e}")
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
    print("Checkpoints have been saved. You can resume later.")