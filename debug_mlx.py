#!/usr/bin/env python3
"""Debug MLX data loading issue."""

import json
from pathlib import Path
import os

print("Current working directory:", os.getcwd())
print("\nChecking data directories:")

# Check various possible locations
locations = [
    "python/sentence_training/data/train.jsonl",
    "python/sentence_training/data/train.json",
    "python/sentence_training/train.jsonl",
    "python/sentence_training/train.json",
    "data/train.jsonl",
    "data/train.json"
]

for loc in locations:
    path = Path(loc)
    print(f"{loc}: {'EXISTS' if path.exists() else 'NOT FOUND'}")

# Check what mlx_lm expects
print("\n" + "="*50)
print("Checking MLX dataset structure...")

try:
    import mlx_lm.tuner.datasets as datasets
    print("\nMLX dataset module loaded successfully")
    
    # Check the source code to understand what it's looking for
    import inspect
    print("\nload_dataset function signature:")
    print(inspect.signature(datasets.load_dataset))
    
    # Try to understand the data path construction
    from types import SimpleNamespace
    args = SimpleNamespace(data="python/sentence_training")
    data_path = Path(args.data)
    
    print(f"\nData path: {data_path}")
    print(f"Data path exists: {data_path.exists()}")
    
    # Check what files MLX looks for
    for split in ["train", "valid", "test"]:
        for ext in [".jsonl", ".json"]:
            file_path = data_path / f"{split}{ext}"
            print(f"{file_path}: {'EXISTS' if file_path.exists() else 'NOT FOUND'}")
            
            # Also check in data subdirectory
            file_path2 = data_path / "data" / f"{split}{ext}"
            print(f"{file_path2}: {'EXISTS' if file_path2.exists() else 'NOT FOUND'}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()