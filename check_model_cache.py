#!/usr/bin/env python3
"""
Check if MLX models are already cached locally.
"""

import os
from pathlib import Path

def check_mlx_cache():
    """Check for cached MLX models."""
    print("Checking for cached MLX models...")
    print("="*60)
    
    # Common cache locations
    cache_dirs = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / "Library" / "Caches" / "huggingface" / "hub",
        Path.home() / ".mlx" / "models"
    ]
    
    models_found = []
    
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            print(f"\n📁 Checking: {cache_dir}")
            
            # Look for model directories
            for item in cache_dir.iterdir():
                if item.is_dir() and "models--" in item.name:
                    model_name = item.name.replace("models--", "").replace("--", "/")
                    size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / (1024**3)
                    models_found.append((model_name, item, size))
                    print(f"   ✅ Found: {model_name} ({size:.1f} GB)")
    
    if not models_found:
        print("\n❌ No cached MLX models found")
        print("\nModels will be downloaded when you run training.")
    else:
        print(f"\n✅ Found {len(models_found)} cached model(s)")
        print("\nThese models are already downloaded and ready to use:")
        for model, path, size in models_found:
            print(f"- {model} ({size:.1f} GB)")
    
    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    
    print(f"\n💾 Disk space available: {free_gb:.1f} GB")
    if free_gb < 5:
        print("⚠️  Warning: Low disk space. You may need 2-4 GB for model download.")
    
    return models_found

if __name__ == "__main__":
    check_mlx_cache()