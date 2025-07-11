"""
Download and prepare Gemma model for training - automated version.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi, login
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()


def download_gemma_for_training():
    """
    Download Gemma 2B model from Hugging Face for training.
    """
    # Model ID on Hugging Face
    model_id = "google/gemma-2b"
    
    # Local cache directory
    cache_dir = Path(__file__).parent / "models"
    cache_dir.mkdir(exist_ok=True)
    
    print(f"Downloading {model_id} for training...")
    print(f"This may take a few minutes depending on your internet connection...")
    print(f"Model will be cached in: {cache_dir}")
    
    try:
        # Download the model with progress tracking
        model_path = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir),
            local_dir=str(cache_dir / "gemma-2b"),
            # Only download necessary files
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
            "status": "downloaded",
            "download_complete": True
        }
        
        config_path = cache_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model configuration saved to: {config_path}")
        
        return model_path
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        return None


if __name__ == "__main__":
    print("=== Automated Gemma Model Download ===")
    
    # Get token from environment
    token = os.environ.get('HF_TOKEN')
    
    if token:
        try:
            # Login with the token
            login(token=token, add_to_git_credential=False)
            api = HfApi()
            user_info = api.whoami()
            print(f"✅ Logged in as: {user_info['name']}")
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            sys.exit(1)
    else:
        print("❌ HF_TOKEN not found in environment")
        sys.exit(1)
    
    print("\nStarting automatic download...")
    model_path = download_gemma_for_training()
    
    if model_path:
        print("\n✅ Model downloaded successfully!")
        print(f"Location: {model_path}")
        print("\nReady for training!")
    else:
        print("\n❌ Download failed")
        sys.exit(1)