"""
Convert bio JSONL data from prompt-completion format to MLX training format.
MLX expects a simpler format for text completion tasks.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def convert_to_mlx_format(data: List[Dict]) -> List[Dict]:
    """
    Convert prompt-completion format to MLX format.
    
    MLX format for text completion:
    {"text": "prompt: <prompt text> completion: <completion text>"}
    
    This format is simpler and works well with MLX's training scripts.
    """
    mlx_data = []
    
    for item in data:
        prompt = item.get('prompt', '').strip()
        completion = item.get('completion', '').strip()
        
        if prompt and completion:
            # MLX format: combine prompt and completion with clear markers
            text = f"prompt: {prompt} completion: {completion}"
            mlx_data.append({"text": text})
    
    return mlx_data

def save_mlx_data(data: List[Dict], output_path: str):
    """Save data in MLX format to JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    input_dir = base_dir / "python" / "mlx_training" / "bio_mlx_improved"
    output_dir = base_dir / "python" / "mlx_server" / "data"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Convert each file
    files_to_convert = ['train.jsonl', 'valid.jsonl', 'test.jsonl']
    
    total_samples = 0
    for filename in files_to_convert:
        input_path = input_dir / filename
        output_path = output_dir / filename
        
        if input_path.exists():
            # Load data
            data = load_jsonl(str(input_path))
            print(f"Loaded {len(data)} samples from {filename}")
            
            # Convert to MLX format
            mlx_data = convert_to_mlx_format(data)
            
            # Save converted data
            save_mlx_data(mlx_data, str(output_path))
            print(f"Saved {len(mlx_data)} samples to {output_path}")
            
            total_samples += len(mlx_data)
        else:
            print(f"Warning: {input_path} not found")
    
    print(f"\nConversion complete! Total samples: {total_samples}")
    
    # Show a few examples
    if (output_dir / "train.jsonl").exists():
        print("\nExample converted samples:")
        examples = load_jsonl(str(output_dir / "train.jsonl"))[:3]
        for i, example in enumerate(examples, 1):
            print(f"\nExample {i}:")
            print(f"Text: {example['text'][:150]}..." if len(example['text']) > 150 else f"Text: {example['text']}")

if __name__ == "__main__":
    main()