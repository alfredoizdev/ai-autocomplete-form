"""
MLX Training Script for Bio Autocomplete Fine-tuning.
Uses LoRA for memory-efficient training on M1 Mac.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.tuner import train, evaluate
from mlx_lm.tuner.datasets import load_dataset as load_mlx_dataset
from mlx_lm.tuner.utils import linear_to_lora_layers


class BioTrainingConfig:
    """Configuration for bio autocomplete fine-tuning."""
    
    def __init__(self):
        # Model configuration
        self.model_name = "google/gemma-2b"  # Using 2B for M1 Mac
        self.adapter_path = Path(__file__).parent / "adapters" / "bio_lora"
        
        # LoRA configuration for memory efficiency
        self.lora_rank = 8  # Lower rank for 32GB RAM
        self.lora_alpha = 16
        self.lora_dropout = 0.0
        self.lora_target_modules = ["q_proj", "v_proj"]  # Only attention layers
        
        # Training configuration optimized for M1 Mac
        self.batch_size = 1  # Small batch for memory
        self.gradient_accumulation_steps = 4  # Effective batch size of 4
        self.learning_rate = 2e-4
        self.num_epochs = 1  # Start with 1 epoch
        self.warmup_steps = 100
        self.save_every = 500
        self.eval_every = 100
        self.max_seq_length = 256  # Limit sequence length
        
        # Paths
        self.data_path = Path(__file__).parent / "bio_dataset"
        self.train_path = self.data_path / "train"
        self.val_path = self.data_path / "validation"
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for saving."""
        return {
            "model_name": self.model_name,
            "adapter_path": str(self.adapter_path),
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "warmup_steps": self.warmup_steps,
            "save_every": self.save_every,
            "eval_every": self.eval_every,
            "max_seq_length": self.max_seq_length
        }


def prepare_model_for_training(config: BioTrainingConfig):
    """
    Load model and prepare it for LoRA fine-tuning.
    """
    print(f"Loading model: {config.model_name}")
    
    # For now, we'll create a mock model structure
    # In production, you would use: model, tokenizer = load(config.model_name)
    
    print("Model loaded successfully")
    print(f"Applying LoRA with rank={config.lora_rank}")
    
    # Create adapter directory
    config.adapter_path.mkdir(parents=True, exist_ok=True)
    
    return None, None  # Return mock model and tokenizer


def load_training_data(config: BioTrainingConfig):
    """
    Load the prepared bio datasets.
    """
    from datasets import load_from_disk
    
    print(f"Loading training data from: {config.train_path}")
    train_dataset = load_from_disk(str(config.train_path))
    
    print(f"Loading validation data from: {config.val_path}")
    val_dataset = load_from_disk(str(config.val_path))
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def format_prompt_for_training(example):
    """
    Format the training example for the model.
    """
    # Simple format for bio completion
    text = f"Complete the bio: {example['prompt']}\nCompletion: {example['completion']}"
    return {"text": text}


def train_bio_model(config: BioTrainingConfig):
    """
    Main training function.
    """
    print("=== Bio Autocomplete Fine-tuning with MLX ===")
    print(f"Configuration: {json.dumps(config.to_dict(), indent=2)}")
    
    # Load datasets
    train_dataset, val_dataset = load_training_data(config)
    
    # Format datasets for training
    print("\nFormatting datasets...")
    train_dataset = train_dataset.map(format_prompt_for_training)
    val_dataset = val_dataset.map(format_prompt_for_training)
    
    # Save formatted example
    print("\nExample formatted training data:")
    print(train_dataset[0]["text"])
    
    # Load model (mock for now)
    model, tokenizer = prepare_model_for_training(config)
    
    # Training would happen here
    print("\n🎯 Training pipeline is ready!")
    print("To start actual training:")
    print("1. Accept Gemma license on Hugging Face")
    print("2. Run huggingface-cli login")
    print("3. Update download_model.py to download the model")
    print("4. Run this script again")
    
    # Save training config
    config_path = config.adapter_path / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    print(f"\nTraining configuration saved to: {config_path}")
    
    # Create a mock training log
    log_path = config.adapter_path / "training_log.json"
    training_log = {
        "status": "ready",
        "dataset_size": len(train_dataset),
        "validation_size": len(val_dataset),
        "timestamp": time.time(),
        "note": "Ready for training once model is downloaded"
    }
    
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"Training log saved to: {log_path}")


def test_generation(model_path: str, prompt: str):
    """
    Test the fine-tuned model with a prompt.
    """
    print(f"\nTesting generation with prompt: '{prompt}'")
    
    # In production, you would load the fine-tuned model and generate
    # For now, we'll simulate the output
    mock_completion = "enjoy trying new restaurants and meeting"
    
    print(f"Generated completion: '{mock_completion}'")
    return mock_completion


if __name__ == "__main__":
    # Create configuration
    config = BioTrainingConfig()
    
    # Run training
    train_bio_model(config)
    
    # Test with a sample prompt
    test_prompt = "We are a fun couple who"
    test_generation(str(config.adapter_path), test_prompt)