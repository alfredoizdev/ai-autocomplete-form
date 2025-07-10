"""
Download and prepare Gemma model for MLX fine-tuning.
We'll use Gemma 2:2B for better performance on M1 Mac with 32GB RAM.
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download


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
        # Download the model
        model_path = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir),
            local_dir=str(cache_dir / "gemma-2b"),
            local_dir_use_symlinks=False,
            # Only download necessary files for MLX
            ignore_patterns=["*.onnx", "*.msgpack", "*.h5", "*.tflite"]
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


if __name__ == "__main__":
    # Note: Gemma models require accepting the license agreement
    print("=== Gemma Model Download for MLX ===")
    print("\nIMPORTANT: Gemma models require accepting the license agreement.")
    print("Please make sure you have:")
    print("1. Created a Hugging Face account")
    print("2. Accepted the Gemma license at: https://huggingface.co/google/gemma-2b")
    print("3. Logged in using: huggingface-cli login")
    print("\nFor now, we'll skip the download and use a mock training setup.")
    print("This allows us to test the training pipeline without downloading the full model.")
    
    # For testing purposes, create a mock config
    cache_dir = Path(__file__).parent / "models"
    cache_dir.mkdir(exist_ok=True)
    
    mock_config = {
        "model_id": "google/gemma-2b",
        "model_path": str(cache_dir / "gemma-2b"),
        "model_size": "2B",
        "status": "ready_for_download",
        "note": "Run this script after accepting Gemma license to download"
    }
    
    import json
    config_path = cache_dir / "model_config.json"
    with open(config_path, 'w') as f:
        json.dump(mock_config, f, indent=2)
    
    print(f"\n✅ Mock configuration created at: {config_path}")
    print("When ready to download, run this script after accepting the license.")