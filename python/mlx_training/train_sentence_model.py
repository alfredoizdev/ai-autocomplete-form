#!/usr/bin/env python3
"""
Train a sentence completion model using MLX with the new high-quality dataset.
This should produce natural, diverse completions without repetitive patterns.
"""

import subprocess
import sys
from pathlib import Path
import json
import time
import os

def check_mlx_installed():
    """Check if MLX is installed."""
    try:
        import mlx
        return True
    except ImportError:
        print("MLX not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mlx-lm"])
        return True

def download_model_if_needed():
    """Check if the base model exists, download if not."""
    print("\nChecking for base model...")
    
    # The model will be cached by MLX automatically
    model_name = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    
    print(f"Using model: {model_name}")
    print("Note: MLX will automatically download the model on first use if needed.")
    print("This may take a few minutes for the first run.\n")
    
    return True

def prepare_training_environment():
    """Set up the training environment."""
    base_dir = Path(__file__).parent.parent.parent
    
    # Create models directory
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Check if training data exists
    train_file = base_dir / "python/sentence_training/mlx/train.jsonl"
    if not train_file.exists():
        print(f"❌ Error: Training data not found at {train_file}")
        print("Please run the data preparation scripts first.")
        return False
    
    # Check data quality
    print("Checking training data quality...")
    with open(train_file, 'r') as f:
        samples = [json.loads(line) for line in f.readlines()[:10]]
    
    print(f"\nSample training examples:")
    for i, sample in enumerate(samples[:3], 1):
        text = sample['text']
        # Extract prompt and completion for display
        if '<|user|>' in text and '<|assistant|>' in text:
            prompt_start = text.find('<|user|>') + len('<|user|>\n')
            prompt_end = text.find('<|end|>')
            prompt = text[prompt_start:prompt_end]
            
            comp_start = text.find('<|assistant|>') + len('<|assistant|>\n')
            comp_end = text.rfind('<|end|>')
            completion = text[comp_start:comp_end]
            
            print(f"\n{i}. Prompt: '{prompt}'")
            print(f"   Completion: '{completion}'")
    
    return True

def run_training():
    """Run the MLX training."""
    base_dir = Path(__file__).parent.parent.parent
    config_path = base_dir / "python/sentence_training/mlx/config_fixed.yaml"
    
    # Change to the base directory so relative paths work
    os.chdir(base_dir)
    print(f"Working directory: {os.getcwd()}")
    
    print("\n" + "="*60)
    print("STARTING MLX TRAINING")
    print("="*60)
    print(f"Config: {config_path}")
    print(f"Output: models/bio-sentence-llama3-lora/")
    print("\nTraining parameters:")
    print("- Model: Llama-3.2-3B-Instruct-4bit")
    print("- Dataset: 4,144 high-quality sentence pairs")
    print("- Iterations: 1,000")
    print("- Batch size: 4")
    print("- LoRA rank: 16")
    print("="*60)
    
    # Run the training command
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--config", str(config_path)
    ]
    
    print(f"\nRunning command: {' '.join(cmd)}")
    print("\nThis will take approximately 10-20 minutes on Apple Silicon...\n")
    
    try:
        # Run training
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 universal_newlines=True, bufsize=1)
        
        # Stream output
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print("\n✅ Training completed successfully!")
            return True
        else:
            print(f"\n❌ Training failed with return code: {process.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        return False

def test_model():
    """Test the trained model with some prompts."""
    print("\n" + "="*60)
    print("TESTING TRAINED MODEL")
    print("="*60)
    
    base_dir = Path(__file__).parent.parent.parent
    adapter_path = base_dir / "models/bio-sentence-llama3-lora"
    
    if not adapter_path.exists():
        print("❌ Model not found. Training may have failed.")
        return
    
    # Test prompts
    test_prompts = [
        "i am a young male swinger looking for",
        "we are a couple interested in",
        "looking for fun people who",
        "i enjoy meeting new",
        "we want to find couples that"
    ]
    
    print("\nTesting with sample prompts...")
    print("(Note: You'll need to update your MLX server to use this model)\n")
    
    for prompt in test_prompts:
        print(f"Prompt: '{prompt}'")
        print("Expected: Natural, varied completions")
        print("(Run the MLX server with the new model to test)\n")
    
    print("To use this model:")
    print("1. Update mlx_model_server.py to point to 'models/bio-sentence-llama3-lora'")
    print("2. Restart the MLX server")
    print("3. Test with the app to see improved completions")

def main():
    """Main training pipeline."""
    print("Bio Sentence Completion Model Training")
    print("="*40)
    
    # Check MLX installation
    if not check_mlx_installed():
        return
    
    # Check for model
    if not download_model_if_needed():
        return
    
    # Prepare environment
    if not prepare_training_environment():
        return
    
    # Confirm before training
    print("\n" + "="*60)
    print("READY TO START TRAINING")
    print("="*60)
    print("This will train a Llama-3.2-3B model on your sentence completion data.")
    print("The training will produce natural, diverse completions.")
    print("\nEstimated time: 10-20 minutes on M1/M2/M3 Mac")
    print("="*60)
    
    response = input("\nProceed with training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Run training
    if run_training():
        # Test the model
        test_model()
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("1. The model has been saved to: models/bio-sentence-llama3-lora/")
        print("2. Update your MLX server to use this new model")
        print("3. Restart the server and test the autocomplete")
        print("4. You should see natural, diverse completions!")
    else:
        print("\nTraining failed. Please check the error messages above.")

if __name__ == "__main__":
    main()