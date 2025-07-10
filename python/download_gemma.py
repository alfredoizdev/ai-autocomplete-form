"""
Download Gemma 2B model for MLX training.
This script handles the download with progress tracking.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def download_gemma_weights():
    """Download Gemma 2B model weights."""
    
    # Check authentication
    token = os.environ.get('HF_TOKEN')
    if not token:
        print("❌ HF_TOKEN not found. Please set it in .env file")
        return False
    
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        api = HfApi()
        user = api.whoami()
        print(f"✅ Authenticated as: {user['name']}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False
    
    # Model details
    model_id = "google/gemma-2b"
    cache_dir = Path("mlx_training/models")
    local_dir = cache_dir / "gemma-2b"
    
    print(f"\n📥 Downloading Gemma 2B model weights...")
    print(f"Destination: {local_dir}")
    
    # Essential files for MLX
    required_files = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    
    # Download each file
    for filename in required_files:
        print(f"\nDownloading {filename}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                cache_dir=str(cache_dir),
                local_dir=str(local_dir),
                resume_download=True,
                token=token
            )
            print(f"✅ Downloaded: {filename}")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            return False
    
    print("\n✅ All model files downloaded successfully!")
    print(f"Model location: {local_dir}")
    
    # Verify download
    weight_files = list(local_dir.glob("*.safetensors"))
    print(f"\nFound {len(weight_files)} weight files:")
    for f in weight_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name}: {size_mb:.1f} MB")
    
    return True

if __name__ == "__main__":
    print("=== Gemma 2B Model Download ===")
    
    if download_gemma_weights():
        print("\n✅ Download complete!")
        print("\nNext steps:")
        print("1. Run: python mlx_training/train_mlx.py")
        print("2. Training will take 2-3 hours")
        print("3. Monitor memory usage in Activity Monitor")
    else:
        print("\n❌ Download failed. Please check your connection and try again.")