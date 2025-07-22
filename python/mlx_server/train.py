#!/usr/bin/env python3
"""
MLX LoRA fine-tuning script for bio autocomplete.
Optimized for M1 Max with 32GB RAM.
"""

import subprocess
import sys
import os
from pathlib import Path
import yaml
import time

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_model_downloaded(model_name):
    """Check if model exists locally, download if needed."""
    print(f"Checking for model: {model_name}")
    
    # MLX will automatically download the model if it doesn't exist
    # We'll use a simple check to see if we can access it
    try:
        from mlx_lm import load
        print(f"Attempting to load {model_name}...")
        # This will download if not present
        model, tokenizer = load(model_name)
        print(f"✅ Model {model_name} is ready")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def run_training(config):
    """Run MLX LoRA fine-tuning."""
    
    # Build the command
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", config['model'],
        "--train",
        "--data", str(Path(__file__).parent / "data"),  # Data directory
        "--batch-size", str(config['training']['batch_size']),
        "--num-layers", str(config['lora']['rank']),
        "--iters", str(config['training']['max_steps']),
        "--learning-rate", str(config['training']['learning_rate']),
        "--save-every", str(config['evaluation']['save_steps']),
        "--test"  # Also run evaluation
    ]
    
    # Add adapter file path
    adapter_path = Path(config['output']['model_dir'])
    adapter_path.mkdir(parents=True, exist_ok=True)
    cmd.extend(["--adapter-path", str(adapter_path)])
    
    print("Starting MLX LoRA training...")
    print(f"Command: {' '.join(cmd)}")
    print("\nTraining Progress:")
    print("-" * 50)
    
    # Run the training
    start_time = time.time()
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line.rstrip())
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code == 0:
            elapsed_time = time.time() - start_time
            print("\n" + "=" * 50)
            print(f"✅ Training completed successfully!")
            print(f"Total time: {elapsed_time/60:.1f} minutes")
            print(f"Model saved to: {adapter_path}")
        else:
            print(f"\n❌ Training failed with return code: {return_code}")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        process.terminate()
        process.wait()
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        
def main():
    """Main training function."""
    print("🚀 MLX Bio Autocomplete Training")
    print("=" * 50)
    
    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(config_path)
    
    print(f"Model: {config['model']}")
    print(f"LoRA Rank: {config['lora']['rank']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    print(f"Max Steps: {config['training']['max_steps']}")
    print("=" * 50)
    
    # Ensure model is available
    if not ensure_model_downloaded(config['model']):
        print("Failed to load model. Please check your configuration.")
        return
    
    # Check data files exist
    data_dir = Path(__file__).parent / "data"
    for split in ['train', 'valid', 'test']:
        data_file = data_dir / f"{split}.jsonl"
        if not data_file.exists():
            print(f"❌ Missing data file: {data_file}")
            print("Please run convert_to_mlx_format.py first")
            return
        else:
            # Count lines in file
            with open(data_file, 'r') as f:
                count = sum(1 for _ in f)
            print(f"✅ {split}.jsonl: {count} samples")
    
    print("\n" + "=" * 50)
    print("Starting training...")
    
    # Run training
    run_training(config)

if __name__ == "__main__":
    main()