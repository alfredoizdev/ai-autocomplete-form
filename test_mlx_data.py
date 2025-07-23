#!/usr/bin/env python3
"""Test if MLX can load our training data."""

import json
from pathlib import Path

# Check data format
data_file = Path("python/sentence_training/mlx/train.jsonl")

print(f"Checking {data_file}...")
print(f"File exists: {data_file.exists()}")
print(f"File size: {data_file.stat().st_size} bytes")

# Read first few lines
with open(data_file, 'r') as f:
    lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    
    # Check first 3 samples
    print("\nFirst 3 samples:")
    for i, line in enumerate(lines[:3]):
        sample = json.loads(line)
        print(f"\nSample {i+1}:")
        print(f"Keys: {list(sample.keys())}")
        print(f"Text length: {len(sample['text'])}")
        print(f"Text preview: {sample['text'][:100]}...")

# Try loading with MLX's dataset loader
print("\n" + "="*50)
print("Testing MLX dataset loading...")

try:
    from mlx_lm.tuner.datasets import load_dataset
    from transformers import AutoTokenizer
    import types
    
    # Create args namespace
    args = types.SimpleNamespace(
        train="python/sentence_training/mlx/train.jsonl",
        valid="python/sentence_training/mlx/validation.jsonl",
        test="python/sentence_training/mlx/test.jsonl",
        max_seq_length=256,
        train_on_completions=True
    )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mlx-community/Llama-3.2-3B-Instruct-4bit")
    
    # Try to load dataset
    print("Loading datasets...")
    train_set, valid_set, test_set = load_dataset(args, tokenizer)
    
    print(f"\n✅ Success!")
    print(f"Train set size: {len(train_set)}")
    print(f"Valid set size: {len(valid_set)}")
    print(f"Test set size: {len(test_set)}")
    
except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()