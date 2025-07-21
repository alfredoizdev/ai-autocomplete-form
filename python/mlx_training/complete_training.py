#!/usr/bin/env python3
"""Complete the training to exactly 1000 iterations."""

import subprocess
import sys
from pathlib import Path

def get_current_iteration():
    """Find the highest iteration checkpoint."""
    adapter_dir = Path("adapters/bio_mistral_fresh")
    checkpoints = list(adapter_dir.glob("*_adapters.safetensors"))
    if not checkpoints:
        return 0
    
    # Get iteration numbers
    iterations = [int(cp.stem.split('_')[0]) for cp in checkpoints]
    return max(iterations)

def main():
    current_iter = get_current_iteration()
    target_iter = 1000
    
    print(f"🎯 MLX Training Completion")
    print("=" * 60)
    print(f"Current iteration: {current_iter}")
    print(f"Target iteration: {target_iter}")
    print(f"Remaining: {target_iter - current_iter}")
    print("=" * 60)
    
    if current_iter >= target_iter:
        print("✅ Training already complete!")
        return
    
    # Build command
    checkpoint_file = f"adapters/bio_mistral_fresh/{current_iter:07d}_adapters.safetensors"
    
    cmd = [
        sys.executable,
        "-m", "mlx_lm", "lora",
        "--config", "training_config_fresh.yaml",
        "--train",
        "--resume-adapter-file", checkpoint_file,
        "--iters", str(target_iter),
        "--save-every", "50"  # Save more frequently near the end
    ]
    
    print(f"\n🚀 Resuming from iteration {current_iter}...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        subprocess.run(cmd, check=True)
        print("\n🎉 Training completed successfully!")
        print(f"✅ Model trained for {target_iter} iterations")
        print("✅ Ready for production use!")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()