"""
Prepare bio data for MLX fine-tuning.
This script converts bio.json into prompt-completion pairs for training.
"""

import json
import random
import sys
from pathlib import Path
from datasets import Dataset

# Add parent directory to path to access data
sys.path.append(str(Path(__file__).parent.parent.parent))


def prepare_bio_training_data(bio_file_path, output_dir):
    """
    Prepare bio data for MLX fine-tuning by creating prompt-completion pairs.
    
    Args:
        bio_file_path: Path to bio.json file
        output_dir: Directory to save the prepared datasets
    """
    # Load bio data
    with open(bio_file_path, 'r') as f:
        bios = json.load(f)
    
    print(f"Loaded {len(bios)} bios from {bio_file_path}")
    
    training_examples = []
    
    # Create training examples from each bio
    for bio_idx, bio_text in enumerate(bios):
        # Skip very short bios
        if len(bio_text) < 20:
            continue
        
        words = bio_text.split()
        
        # Create multiple examples from each bio
        # Starting from 3 words and creating examples with 5-word completions
        for i in range(3, len(words) - 5, 3):
            prompt = " ".join(words[:i])
            completion = " ".join(words[i:i+5])
            
            # Create training example in the format MLX expects
            training_examples.append({
                "prompt": prompt,
                "completion": completion,
                "bio_id": bio_idx  # Track which bio this came from
            })
    
    print(f"Created {len(training_examples)} training examples")
    
    # Shuffle for better training
    random.seed(42)  # Set seed for reproducibility
    random.shuffle(training_examples)
    
    # Split into train/validation (90/10 split)
    split_idx = int(0.9 * len(training_examples))
    train_data = training_examples[:split_idx]
    val_data = training_examples[split_idx:]
    
    print(f"Train set: {len(train_data)} examples")
    print(f"Validation set: {len(val_data)} examples")
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Save datasets
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_path = output_path / "train"
    val_path = output_path / "validation"
    
    train_dataset.save_to_disk(str(train_path))
    val_dataset.save_to_disk(str(val_path))
    
    print(f"\nDatasets saved to:")
    print(f"  Training: {train_path}")
    print(f"  Validation: {val_path}")
    
    # Save some example pairs for verification
    print("\nExample training pairs:")
    for i in range(min(3, len(train_data))):
        example = train_data[i]
        print(f"\nExample {i+1}:")
        print(f"  Prompt: '{example['prompt']}'")
        print(f"  Completion: '{example['completion']}'")
    
    # Save dataset statistics
    stats = {
        "total_bios": len(bios),
        "total_examples": len(training_examples),
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "avg_prompt_length": sum(len(ex["prompt"].split()) for ex in training_examples) / len(training_examples),
        "avg_completion_length": sum(len(ex["completion"].split()) for ex in training_examples) / len(training_examples)
    }
    
    stats_path = output_path / "dataset_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset statistics saved to: {stats_path}")
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    # Path to bio.json file
    bio_file = Path(__file__).parent.parent.parent / "data" / "bio.json"
    
    # Output directory for datasets
    output_dir = Path(__file__).parent / "bio_dataset"
    
    # Prepare the data
    prepare_bio_training_data(bio_file, output_dir)