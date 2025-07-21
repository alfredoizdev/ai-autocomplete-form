#!/usr/bin/env python3
"""Automatically continue training with retries."""

import subprocess
import sys
import time
from pathlib import Path

def run_training_with_retries(max_retries=5, target_iters=2000):
    """Run training with automatic retries on failure."""
    
    config_path = "training_config_fresh.yaml"
    adapter_dir = Path("adapters/bio_mistral_fresh")
    
    for attempt in range(max_retries):
        print(f"\n🔄 Training attempt {attempt + 1}/{max_retries}")
        print("-" * 60)
        
        # Find latest checkpoint
        checkpoints = list(adapter_dir.glob("*_adapters.safetensors"))
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[0]))
            latest_checkpoint = checkpoints[-1]
            current_iter = int(latest_checkpoint.stem.split('_')[0])
            
            if current_iter >= target_iters:
                print(f"✅ Training complete! Reached {current_iter} iterations")
                return True
            
            print(f"📍 Resuming from iteration {current_iter}")
            resume_args = ["--resume-adapter-file", str(latest_checkpoint)]
        else:
            print("🆕 Starting fresh training")
            resume_args = []
        
        # Build command
        cmd = [
            sys.executable,
            "-m", "mlx_lm", "lora",
            "--config", config_path,
            "--train",
            "--iters", str(target_iters)
        ] + resume_args
        
        try:
            # Run training
            result = subprocess.run(cmd, check=False)
            
            if result.returncode == 0:
                print("✅ Training completed successfully!")
                return True
            else:
                print(f"⚠️  Training exited with code {result.returncode}")
                
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Wait before retry
        if attempt < max_retries - 1:
            print(f"⏳ Waiting 10 seconds before retry...")
            time.sleep(10)
    
    print("\n❌ Maximum retries reached")
    return False

def main():
    print("🤖 MLX Auto-Training Script")
    print("=" * 60)
    print("This script will automatically continue training")
    print("and retry if interrupted.")
    print("-" * 60)
    
    success = run_training_with_retries()
    
    if success:
        print("\n🎉 Training complete!")
        print("Run test_fresh_model.py to test the final model")
    else:
        print("\n⚠️  Training incomplete")
        print("Check logs and try again")

if __name__ == "__main__":
    main()