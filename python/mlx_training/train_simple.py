"""
Simple training script for bio autocomplete using transformers library.
Optimized for M1 Mac with 32GB RAM.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

# Disable tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk
import torch

# Use MPS (Metal Performance Shaders) for M1 Mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


class SimpleTrainingConfig:
    """Simple training configuration."""
    
    def __init__(self):
        self.model_path = Path(__file__).parent / "models" / "gemma-2b"
        self.output_dir = Path(__file__).parent / "simple_output"
        self.data_path = Path(__file__).parent / "bio_dataset"
        
        # Very conservative settings for 32GB RAM
        self.batch_size = 1
        self.gradient_accumulation_steps = 8
        self.learning_rate = 5e-5
        self.num_train_epochs = 0.1  # Just 10% of one epoch for demo
        self.max_seq_length = 64  # Very short sequences
        self.fp16 = False  # MPS doesn't support fp16 well
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)


def load_and_prepare_model(config):
    """Load model in 8-bit for memory efficiency."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(config.model_path))
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model (this may take a minute)...")
    
    # Load model with reduced memory footprint
    model = AutoModelForCausalLM.from_pretrained(
        str(config.model_path),
        torch_dtype=torch.float32,  # MPS works better with float32
        low_cpu_mem_usage=True,
        device_map={"": device}
    )
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # Freeze most layers, only train last 2
    for name, param in model.named_parameters():
        param.requires_grad = False
        if "layer.16" in name or "layer.17" in name or "norm" in name:
            param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Model loaded!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return model, tokenizer


def prepare_datasets(config, tokenizer):
    """Prepare training data."""
    print("\nLoading datasets...")
    train_dataset = load_from_disk(str(config.data_path / "train"))
    val_dataset = load_from_disk(str(config.data_path / "validation"))
    
    # Use only a small subset for demo
    train_dataset = train_dataset.select(range(min(100, len(train_dataset))))
    val_dataset = val_dataset.select(range(min(20, len(val_dataset))))
    
    print(f"Using {len(train_dataset)} training examples (subset)")
    print(f"Using {len(val_dataset)} validation examples (subset)")
    
    def tokenize_function(examples):
        # Simple format: prompt + completion
        texts = [f"Complete: {p} → {c}" for p, c in zip(examples["prompt"], examples["completion"])]
        
        return tokenizer(
            texts,
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length"
        )
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)
    
    return train_dataset, val_dataset


def train_model(model, tokenizer, train_dataset, val_dataset, config):
    """Train the model."""
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        overwrite_output_dir=True,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_steps=10,
        save_steps=10,
        warmup_steps=10,
        logging_steps=5,
        learning_rate=config.learning_rate,
        fp16=config.fp16,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=1,
        report_to="none",  # Disable wandb, etc.
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Important for MPS
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    # Train
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.1f} seconds")
    
    # Save the model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Save training info
    training_info = {
        "base_model": "google/gemma-2b",
        "training_time_seconds": training_time,
        "examples_trained": len(train_dataset),
        "trainable_params_pct": f"{trainable_params/total_params*100:.2f}%",
        "device": str(device),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(config.output_dir / "training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    
    return trainer


def test_generation(model, tokenizer):
    """Test the fine-tuned model."""
    print("\n" + "="*50)
    print("Testing model generation...")
    print("="*50)
    
    test_prompts = [
        "We are a couple who",
        "Looking for friends to",
        "My hobbies include"
    ]
    
    model.eval()
    for prompt in test_prompts:
        inputs = tokenizer(f"Complete: {prompt} →", return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated.split("→")[-1].strip()
        
        print(f"\nPrompt: '{prompt}'")
        print(f"Completion: '{completion}'")


def main():
    """Main training function."""
    print("=== Simple Bio Autocomplete Training ===")
    print(f"Device: {device}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    config = SimpleTrainingConfig()
    
    # Load model
    model, tokenizer = load_and_prepare_model(config)
    
    # Prepare data
    train_dataset, val_dataset = prepare_datasets(config, tokenizer)
    
    # Train
    trainer = train_model(model, tokenizer, train_dataset, val_dataset, config)
    
    # Test
    test_generation(model, tokenizer)
    
    print("\n✅ Training complete!")
    print(f"Model saved to: {config.output_dir}")
    print("\nNext steps:")
    print("1. Convert to GGUF format for Ollama")
    print("2. Or use directly with transformers library")


if __name__ == "__main__":
    # Check if MPS is available
    if not torch.backends.mps.is_available():
        print("⚠️  MPS (Metal Performance Shaders) not available")
        print("Training will use CPU (slower)")
        response = input("\nContinue with CPU training? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            exit()
    
    main()