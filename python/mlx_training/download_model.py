"""
Download and prepare Gemma model for MLX fine-tuning.
We'll use Gemma 2:2B for better performance on M1 Mac with 32GB RAM.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi, login
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def download_gemma_for_mlx():
    """
    Download Gemma 2B model from Hugging Face for MLX training.
    Using the smaller 2B model for better memory efficiency on M1 Mac.
    """
    # Model ID on Hugging Face
    model_id = "google/gemma-2b"  # Using the 2B version for M1 Mac
    
    # Local cache directory
    cache_dir = Path(__file__).parent / "models"
    cache_dir.mkdir(exist_ok=True)
    
    print(f"Downloading {model_id} for MLX training...")
    print(f"This may take a few minutes depending on your internet connection...")
    print(f"Model will be cached in: {cache_dir}")
    
    try:
        # Download the model with progress tracking
        model_path = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir),
            local_dir=str(cache_dir / "gemma-2b"),
            # Only download necessary files for MLX
            ignore_patterns=["*.onnx", "*.msgpack", "*.h5", "*.tflite"],
            resume_download=True,  # Resume if interrupted
            max_workers=2  # Limit concurrent downloads
        )
        
        print(f"\n✅ Model downloaded successfully to: {model_path}")
        
        # Create a config file to track the download
        config = {
            "model_id": model_id,
            "model_path": str(model_path),
            "model_size": "2B",
            "download_date": str(Path(model_path).stat().st_mtime)
        }
        
        import json
        config_path = cache_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model configuration saved to: {config_path}")
        
        return model_path
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have an active internet connection")
        print("2. You may need to login to Hugging Face:")
        print("   huggingface-cli login")
        print("3. Accept the model license at:")
        print(f"   https://huggingface.co/{model_id}")
        return None


def check_hf_login():
    """Check if user is logged into Hugging Face."""
    # Try to get token from environment
    token = os.environ.get('HF_TOKEN')
    
    if token:
        try:
            # Login with the token
            login(token=token, add_to_git_credential=False)
            api = HfApi()
            user_info = api.whoami()
            print(f"✅ Logged in as: {user_info['name']}")
            return True
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False
    else:
        print("❌ HF_TOKEN not found")
        print("\nPlease set your token:")
        print("1. Create a .env file in the python directory")
        print("2. Add: HF_TOKEN=your_token_here")
        print("3. Or set environment variable: export HF_TOKEN='your_token'")
        return False


if __name__ == "__main__":
    print("=== Gemma Model Download for MLX ===")
    
    # Check if logged in
    if not check_hf_login():
        sys.exit(1)
    
    print("\n✅ You have accepted the Gemma license")
    print("Ready to download Gemma 2B model")
    
    # Ask for confirmation
    response = input("\nProceed with download? This will download ~2-3GB. (y/n): ")
    
    if response.lower() == 'y':
        print("\nStarting download...")
        model_path = download_gemma_for_mlx()
        
        if model_path:
            print("\n✅ Model downloaded successfully!")
            print(f"Location: {model_path}")
            print("\nNext step: Run python mlx_training/train_mlx.py")
    else:
        print("\nDownload cancelled.")
        print("Run this script again when ready to download.")