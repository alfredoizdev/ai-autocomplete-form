#!/usr/bin/env python3
"""
Download and prepare the base model for MLX training.
"""

import subprocess
import sys
from pathlib import Path
import os

def install_dependencies():
    """Install required packages."""
    print("Checking dependencies...")
    packages = ["mlx-lm", "huggingface-hub"]
    
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed")

def download_model():
    """Download the base model using MLX."""
    from mlx_lm import load
    
    # Change to base directory
    base_dir = Path(__file__).parent.parent.parent
    os.chdir(base_dir)
    print(f"Working directory: {os.getcwd()}")
    
    print("\n" + "="*60)
    print("DOWNLOADING BASE MODEL")
    print("="*60)
    
    # Model options (in order of recommendation)
    models = [
        {
            "name": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "description": "Llama 3.2 3B - Excellent for natural language",
            "size": "~2GB"
        },
        {
            "name": "mlx-community/Mistral-7B-Instruct-v0.2-4bit", 
            "description": "Mistral 7B - Larger, very capable",
            "size": "~4GB"
        },
        {
            "name": "mlx-community/Phi-3-mini-4k-instruct-4bit",
            "description": "Phi-3 - Current model (causes repetition)",
            "size": "~2GB"
        }
    ]
    
    print("Available models:")
    for i, model in enumerate(models, 1):
        print(f"\n{i}. {model['name']}")
        print(f"   {model['description']}")
        print(f"   Size: {model['size']}")
    
    print("\nRecommendation: Use option 1 (Llama-3.2-3B) for best results")
    
    choice = input("\nSelect model (1-3) [default: 1]: ").strip() or "1"
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            selected_model = models[idx]["name"]
        else:
            print("Invalid choice, using default")
            selected_model = models[0]["name"]
    except ValueError:
        print("Invalid input, using default")
        selected_model = models[0]["name"]
    
    print(f"\n✅ Selected: {selected_model}")
    print("\nDownloading model... This may take a few minutes.")
    print("The model will be cached for future use.\n")
    
    try:
        # Load the model (this downloads it if not cached)
        print(f"Loading {selected_model}...")
        model, tokenizer = load(selected_model)
        
        print("\n✅ Model downloaded and loaded successfully!")
        
        # Test the model
        print("\nTesting model with a simple prompt...")
        prompt = "Complete this sentence: I am looking for"
        
        # Create a simple test to verify model works
        from mlx_lm import generate
        
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=20,
            verbose=False
        )
        
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print("\n✅ Model is working correctly!")
        
        # Update the config file with the selected model
        update_config(selected_model)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\n" + "="*60)
        print("MANUAL DOWNLOAD INSTRUCTIONS")
        print("="*60)
        print(f"\nTo manually download {selected_model}:")
        print("\n1. Using Hugging Face CLI:")
        print(f"   huggingface-cli download {selected_model}")
        print("\n2. Using Python:")
        print("   python3 -c \"from mlx_lm import load; load('{}')\".format(selected_model)")
        print("\n3. Direct download:")
        print(f"   Visit: https://huggingface.co/{selected_model}")
        print("   Download all files to: ~/.cache/huggingface/hub/")
        print("\n" + "="*60)
        print("\nPossible issues:")
        print("- Internet connection problems")
        print("- Insufficient disk space (need ~2-4GB)")
        print("- Hugging Face server issues")
        print("- Model access restrictions")
        
        response = input("\nWould you like to try a different model? (y/n): ")
        if response.lower() == 'y':
            return download_model()  # Recursive call to try again
        
        print("\nPausing for manual download...")
        print("After downloading manually, run this script again.")
        return False

def update_config(model_name):
    """Update the training config with the selected model."""
    base_dir = Path(__file__).parent.parent.parent
    config_path = base_dir / "python/sentence_training/mlx/config.yaml"
    
    if config_path.exists():
        print(f"\nUpdating config to use {model_name}...")
        
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Update the model line
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('model:'):
                lines[i] = f'model: "{model_name}"'
                break
        
        with open(config_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print("✅ Config updated")

def main():
    """Main function."""
    print("MLX Model Download Tool")
    print("="*60)
    
    # Install dependencies
    install_dependencies()
    
    # Download model
    if download_model():
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("1. Model is ready for training")
        print("2. Run: ./start_sentence_training.sh")
        print("3. Training will use your high-quality sentence data")
        print("4. Expected time: 10-20 minutes on Apple Silicon")
    else:
        print("\nModel download failed. Please check the errors above.")

if __name__ == "__main__":
    main()