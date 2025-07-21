#!/usr/bin/env python3
"""Monitor training progress and continue from latest checkpoint."""

import subprocess
import sys
import os
from pathlib import Path
import time

def find_latest_checkpoint(adapter_dir):
    """Find the latest checkpoint file."""
    adapter_path = Path(adapter_dir)
    checkpoints = list(adapter_path.glob("*_adapters.safetensors"))
    if not checkpoints:
        return None
    
    # Sort by iteration number
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[0]))
    return checkpoints[-1]

def get_iteration_from_checkpoint(checkpoint_path):
    """Extract iteration number from checkpoint filename."""
    return int(checkpoint_path.stem.split('_')[0])

def continue_training(checkpoint_path, target_iters=2000):
    """Continue training from checkpoint."""
    current_iter = get_iteration_from_checkpoint(checkpoint_path)
    
    print(f"📊 Current Status:")
    print(f"   Latest checkpoint: {checkpoint_path.name}")
    print(f"   Current iteration: {current_iter}")
    print(f"   Target iterations: {target_iters}")
    print(f"   Remaining iterations: {target_iters - current_iter}")
    print("-" * 60)
    
    if current_iter >= target_iters:
        print("✅ Training already complete!")
        return True
    
    cmd = [
        sys.executable,
        "-m", "mlx_lm", "lora",
        "--config", "training_config_fresh.yaml",
        "--train",
        "--resume-adapter-file", str(checkpoint_path),
        "--iters", str(target_iters)
    ]
    
    print(f"🚀 Resuming training from iteration {current_iter}...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def main():
    adapter_dir = "adapters/bio_mistral_fresh"
    target_iters = 2000
    
    print("🔍 MLX Training Monitor")
    print("=" * 60)
    
    # Find latest checkpoint
    latest_checkpoint = find_latest_checkpoint(adapter_dir)
    
    if not latest_checkpoint:
        print("❌ No checkpoints found!")
        return
    
    # Continue training
    success = continue_training(latest_checkpoint, target_iters)
    
    if success:
        print("\n✅ Training completed successfully!")
        print(f"   Final model saved in: {adapter_dir}")
        print(f"   Ready for testing with test_fresh_model.py")
    else:
        # Check for new checkpoint after interruption
        new_checkpoint = find_latest_checkpoint(adapter_dir)
        if new_checkpoint != latest_checkpoint:
            new_iter = get_iteration_from_checkpoint(new_checkpoint)
            print(f"\n💾 Progress saved at iteration {new_iter}")
            print(f"   Run this script again to continue")

if __name__ == "__main__":
    main()