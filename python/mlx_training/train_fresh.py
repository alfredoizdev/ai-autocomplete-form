#!/usr/bin/env python3
"""Fresh training run with optimized parameters for bio autocomplete."""

import subprocess
import sys
import yaml
from pathlib import Path

# Create optimized config for bio training
config = {
    "model": "mlx-community/Mistral-7B-Instruct-v0.2-4bit",
    "train": True,
    "data": "./bio_dataset",
    "seed": 42,
    
    # LoRA configuration
    "lora_layers": 16,
    "lora_parameters": {
        "rank": 16,  # Increased for better quality
        "alpha": 32,  # Alpha = 2 * rank is often good
        "scale": 10.0,  # Lower scale for stability
        "dropout": 0.1,  # Slight dropout for regularization
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
    },
    
    # Training parameters
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-5,  # Lower learning rate for stability
    "warmup_steps": 100,
    "iters": 3000,  # More iterations for better convergence
    
    # Memory optimization
    "max_seq_length": 512,
    "gradient_checkpointing": True,
    
    # Evaluation and checkpointing
    "eval_steps": 200,
    "save_every": 500,
    "test_every": 1000,
    
    # Output
    "adapter_path": "./adapters/bio_mistral_fresh",
}

# Save config
config_path = Path("training_config_fresh.yaml")
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("🚀 Starting fresh training run with optimized parameters")
print("-" * 60)
print(f"Model: {config['model']}")
print(f"LoRA rank: {config['lora_parameters']['rank']}")
print(f"Learning rate: {config['learning_rate']}")
print(f"Total iterations: {config['iters']}")
print(f"Checkpoint every: {config['save_every']} iterations")
print("-" * 60)

# Build command
cmd = [
    sys.executable,
    "-m", "mlx_lm", "lora",
    "--config", str(config_path),
    "--train"
]

print(f"\nCommand: {' '.join(cmd)}")
print("\nExpected training time: 4-5 hours")
print("Expected memory usage: 10-12GB")
print("\nStarting training...")

# Execute
try:
    subprocess.run(cmd, check=True)
    print("\n✅ Training completed successfully!")
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted. Checkpoints saved.")
except Exception as e:
    print(f"\n❌ Error: {e}")