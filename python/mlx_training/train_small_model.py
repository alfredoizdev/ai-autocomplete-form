"""
Train the small GPT2 model on bio data.
Optimized for M1 Mac with 32GB RAM.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk
import torch

# Check device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


class SmallModelTrainingConfig:
    """Training configuration for small model."""
    
    def __init__(self):
        self.model_path = Path(__file__).parent / "small_model"
        self.output_dir = Path(__file__).parent / "trained_small_model"
        self.data_path = Path(__file__).parent / "bio_dataset"
        
        # Training settings optimized for small model
        self.batch_size = 4  # Can use larger batch with small model
        self.gradient_accumulation_steps = 2
        self.learning_rate = 5e-4
        self.num_train_epochs = 2  # Can do more epochs
        self.max_seq_length = 128
        self.warmup_steps = 100
        self.logging_steps = 50
        self.save_steps = 500
        self.eval_steps = 100
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)


def load_model_and_tokenizer(config):
    """Load the small model and tokenizer."""
    print("Loading model and tokenizer...")
    
    model = GPT2LMHeadModel.from_pretrained(
        str(config.model_path),
        torch_dtype=torch.float32,
        device_map={"": device}
    )
    
    tokenizer = GPT2Tokenizer.from_pretrained(str(config.model_path))
    tokenizer.pad_token = tokenizer.eos_token
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded! Total parameters: {total_params:,}")
    
    return model, tokenizer


def prepare_datasets(config, tokenizer):
    """Prepare training datasets."""
    print("\nLoading datasets...")
    
    train_dataset = load_from_disk(str(config.data_path / "train"))
    val_dataset = load_from_disk(str(config.data_path / "validation"))
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    def tokenize_function(examples):
        # Format: "Bio: {prompt} -> {completion}"
        texts = []
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            text = f"Bio: {prompt} -> {completion}"
            texts.append(text)
        
        # Tokenize
        model_inputs = tokenizer(
            texts,
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length"
        )
        
        # Set labels (same as input_ids for causal LM)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs
    
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train set"
    )
    
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True, 
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation set"
    )
    
    return train_dataset, val_dataset


def train_model(model, tokenizer, train_dataset, val_dataset, config):
    """Train the model."""
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
        pad_to_multiple_of=8
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        overwrite_output_dir=True,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        push_to_hub=False,
        logging_dir=str(config.output_dir / "logs"),
        logging_first_step=True,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    print(f"Total training steps: {len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps) * config.num_train_epochs}")
    print("This should take 10-20 minutes on M1 Mac\n")
    
    # Train
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time
    
    print(f"\n✅ Training completed in {training_time/60:.1f} minutes")
    
    # Save model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    metrics["training_time_minutes"] = training_time / 60
    metrics["device"] = str(device)
    
    with open(config.output_dir / "training_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return trainer


def test_generation(model, tokenizer, config):
    """Test the fine-tuned model."""
    print("\n" + "="*50)
    print("Testing fine-tuned model...")
    print("="*50)
    
    test_prompts = [
        "We are a couple who",
        "Looking for friends to",
        "My hobbies include",
        "I enjoy",
        "We love to"
    ]
    
    model.eval()
    
    for prompt in test_prompts:
        # Format prompt
        text = f"Bio: {prompt} ->"
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the completion part
        if "->" in generated:
            completion = generated.split("->")[-1].strip()
        else:
            completion = generated[len(text):].strip()
        
        print(f"\nPrompt: '{prompt}'")
        print(f"Completion: '{completion[:50]}...'")  # Limit output length


def main():
    """Main training function."""
    print("=== Small Model Bio Autocomplete Training ===")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    config = SmallModelTrainingConfig()
    
    # Check if model exists
    if not config.model_path.exists():
        print(f"❌ Model not found at: {config.model_path}")
        print("Please run create_small_model.py first")
        return
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(config, tokenizer)
    
    # Train
    trainer = train_model(model, tokenizer, train_dataset, val_dataset, config)
    
    # Test
    test_generation(model, tokenizer, config)
    
    print("\n" + "="*50)
    print("✅ Training complete!")
    print("="*50)
    print(f"\nTrained model saved to: {config.output_dir}")
    print("\nNext steps:")
    print("1. Test the model more thoroughly")
    print("2. Convert to GGUF format for Ollama")
    print("3. Integrate with your autocomplete system")
    
    # Save final summary
    summary = {
        "model_type": "gpt2-small-custom",
        "training_examples": len(train_dataset),
        "epochs": config.num_train_epochs,
        "output_dir": str(config.output_dir),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(config.output_dir / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()