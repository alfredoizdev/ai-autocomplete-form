"""
Optimized training script for bio autocomplete.
Works with larger dataset and handles memory constraints on M1 Mac.
"""

import os
import json
import time
import torch
from pathlib import Path
from datetime import datetime
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# Disable tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Use MPS (Metal Performance Shaders) for M1 Mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


class BioTrainingConfig:
    """Optimized training configuration for bio autocomplete."""
    
    def __init__(self):
        # Use a smaller model that we can download quickly
        self.model_name = "microsoft/phi-2"  # 2.7B params, works great for autocomplete
        self.output_dir = Path(__file__).parent / "bio_finetuned_model"
        self.data_path = Path(__file__).parent / "bio_dataset"
        
        # Optimized settings for M1 Mac with 32GB RAM
        self.batch_size = 2
        self.gradient_accumulation_steps = 8  # Effective batch size of 16
        self.learning_rate = 2e-4
        self.num_train_epochs = 0.5  # Train on 50% of data for faster results
        self.max_seq_length = 128
        self.warmup_steps = 100
        self.logging_steps = 10
        self.save_steps = 500
        self.eval_steps = 100
        
        # LoRA configuration for memory efficiency
        self.use_lora = True
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)


def load_and_prepare_model(config):
    """Load model with LoRA for memory-efficient training."""
    print(f"Loading {config.model_name}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model with memory optimization
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,  # MPS works better with float32
        trust_remote_code=True,
        device_map={"": device}
    )
    
    if config.use_lora:
        print("Applying LoRA configuration...")
        
        # Prepare model for training
        model = prepare_model_for_kbit_training(model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()
    
    return model, tokenizer


def prepare_datasets(config, tokenizer):
    """Prepare training data with the new larger dataset."""
    print("\nLoading datasets...")
    train_dataset = load_from_disk(str(config.data_path / "train"))
    val_dataset = load_from_disk(str(config.data_path / "validation"))
    
    print(f"Full dataset: {len(train_dataset)} training examples")
    print(f"Full dataset: {len(val_dataset)} validation examples")
    
    # Use full dataset but with shorter epochs
    print(f"Training on full dataset with {config.num_train_epochs} epochs")
    
    def tokenize_function(examples):
        # Format for autocomplete training
        texts = []
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            # Special format for bio autocomplete
            text = f"<|bio|>{prompt}<|complete|>{completion}<|endoftext|>"
            texts.append(text)
        
        return tokenizer(
            texts,
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length"
        )
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=4,  # Use multiple cores
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = val_dataset.map(
        tokenize_function, 
        batched=True,
        num_proc=4,
        remove_columns=val_dataset.column_names
    )
    
    return train_dataset, val_dataset


def train_model(model, tokenizer, train_dataset, val_dataset, config):
    """Train the model with optimized settings."""
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
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
        fp16=False,  # MPS doesn't support fp16 well
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Important for MPS
        optim="adamw_torch",  # Use standard AdamW
        gradient_checkpointing=not config.use_lora,  # Only if not using LoRA
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print("\n" + "="*60)
    print("Starting training with bio autocomplete data...")
    print(f"Total training steps: {len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps) * config.num_train_epochs}")
    print("="*60)
    
    # Train
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    print(f"\n✅ Training completed in {training_time/60:.1f} minutes")
    
    # Save the model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Save training info
    training_info = {
        "base_model": config.model_name,
        "training_time_minutes": training_time/60,
        "examples_trained": len(train_dataset),
        "num_epochs": config.num_train_epochs,
        "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
        "use_lora": config.use_lora,
        "device": str(device),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(config.output_dir / "training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    
    return trainer


def test_generation(model, tokenizer, config):
    """Test the fine-tuned model with bio-specific prompts."""
    print("\n" + "="*60)
    print("Testing fine-tuned model on bio completions...")
    print("="*60)
    
    test_prompts = [
        "We are a couple who",
        "Looking for friends to",
        "I enjoy outdoor activities and",
        "My hobbies include",
        "We are interested in meeting",
        "I am looking for someone who"
    ]
    
    model.eval()
    for prompt in test_prompts:
        # Use the same format as training
        input_text = f"<|bio|>{prompt}<|complete|>"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.encode("<|endoftext|>")[0]
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the completion part
        if "<|complete|>" in generated:
            completion = generated.split("<|complete|>")[1].replace("<|endoftext|>", "").strip()
        else:
            completion = generated[len(input_text):].strip()
        
        print(f"\nPrompt: '{prompt}'")
        print(f"Completion: '{completion}'")


def main():
    """Main training function."""
    print("=== Bio Autocomplete Training (Optimized) ===")
    print(f"Device: {device}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    config = BioTrainingConfig()
    
    try:
        # Load model
        model, tokenizer = load_and_prepare_model(config)
        
        # Prepare data
        train_dataset, val_dataset = prepare_datasets(config, tokenizer)
        
        # Train
        trainer = train_model(model, tokenizer, train_dataset, val_dataset, config)
        
        # Test
        test_generation(model, tokenizer, config)
        
        print("\n✅ Training complete!")
        print(f"Model saved to: {config.output_dir}")
        print("\nNext steps:")
        print("1. Convert model for production use")
        print("2. Test with your autocomplete API")
        print("3. Compare performance with current system")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check dependencies
    try:
        import peft
        print("✅ PEFT library available for LoRA training")
    except ImportError:
        print("⚠️  PEFT not installed. Installing now...")
        os.system("pip install peft==0.13.3")
        print("Please run the script again after installation.")
        exit()
    
    main()