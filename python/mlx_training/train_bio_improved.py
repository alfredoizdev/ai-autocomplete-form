"""
Improved training script for bio autocomplete with proper data formatting.
Fixes the tokenization issues that caused poor model performance.
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
    TrainerCallback
)
from peft import LoraConfig, TaskType, get_peft_model

# Disable tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Use MPS (Metal Performance Shaders) for M1 Mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


class ImprovedBioTrainingConfig:
    """Improved training configuration with better parameters."""
    
    def __init__(self):
        # Use GPT2 (medium size) for better quality
        self.model_name = "gpt2"  # 124M params - better than DistilGPT2's 82M
        self.output_dir = Path(__file__).parent / "bio_gpt2_improved"
        self.data_path = Path(__file__).parent / "bio_dataset"
        
        # Quality-focused settings
        self.batch_size = 2  # Smaller for GPT2
        self.gradient_accumulation_steps = 8  # Effective batch size of 16
        self.learning_rate = 5e-5  # Lower learning rate for stability
        self.num_train_epochs = 2  # Fewer epochs to prevent overfitting
        self.max_seq_length = 128  
        self.warmup_steps = 500  # More warmup for stable training
        self.logging_steps = 50
        self.save_steps = 1000
        self.eval_steps = 500
        
        # LoRA configuration for memory efficiency
        self.use_lora = True
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.lora_target_modules = ["c_attn", "c_proj"]  # GPT2 attention layers
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)


def load_and_prepare_model(config):
    """Load model with LoRA for efficient training."""
    print(f"\nLoading {config.model_name} for improved training...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # IMPORTANT: Set pad token properly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for causal LM
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,  # MPS works better with float32
        device_map={"": device}
    )
    
    # Get model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params/1e6:.1f}M parameters")
    
    if config.use_lora:
        print("Applying LoRA for efficient training...")
        
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
    
    return model, tokenizer


def prepare_datasets(config, tokenizer):
    """Prepare training data with improved formatting."""
    print("\nLoading datasets...")
    train_dataset = load_from_disk(str(config.data_path / "train"))
    val_dataset = load_from_disk(str(config.data_path / "validation"))
    
    print(f"Dataset size: {len(train_dataset)} training examples")
    
    # Use a subset for testing
    if len(train_dataset) > 10000:
        print("Using subset of 10000 examples for faster iteration")
        train_dataset = train_dataset.select(range(10000))
        val_dataset = val_dataset.select(range(1000))
    
    def tokenize_function(examples):
        """Improved tokenization with proper formatting."""
        # Create properly formatted texts
        texts = []
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            # Add a space and completion, then EOS token
            # This helps the model learn where completions should end
            text = f"{prompt} {completion}{tokenizer.eos_token}"
            texts.append(text)
        
        # Tokenize with attention to prompt length
        model_inputs = tokenizer(
            texts,
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
            return_tensors=None
        )
        
        # Create labels (same as input for causal LM)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs
    
    # Tokenize datasets
    print("Tokenizing datasets with improved format...")
    train_dataset = train_dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=4,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train set"
    )
    
    val_dataset = val_dataset.map(
        tokenize_function, 
        batched=True,
        num_proc=4,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation set"
    )
    
    return train_dataset, val_dataset


def train_model(model, tokenizer, train_dataset, val_dataset, config):
    """Train with improved parameters."""
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
        pad_to_multiple_of=8  # Efficiency
    )
    
    # Calculate total steps
    total_steps = len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps) * config.num_train_epochs
    
    # Training arguments with better defaults
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        overwrite_output_dir=True,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        weight_decay=0.01,  # Add weight decay
        adam_epsilon=1e-8,
        max_grad_norm=1.0,  # Gradient clipping
        fp16=False,  # MPS doesn't support fp16 well
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="adamw_torch",
        lr_scheduler_type="cosine",  # Better than linear
        prediction_loss_only=True,
    )
    
    # Progress callback
    class ProgressCallback(TrainerCallback):
        def __init__(self, total_steps):
            self.total_steps = total_steps
            self.start_time = time.time()
            
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.global_step > 0 and state.global_step % args.logging_steps == 0:
                elapsed = time.time() - self.start_time
                steps_per_sec = state.global_step / elapsed
                eta = (self.total_steps - state.global_step) / steps_per_sec
                
                loss = logs.get('loss', 'N/A') if logs else 'N/A'
                lr = logs.get('learning_rate', 'N/A') if logs else 'N/A'
                
                print(f"\nStep {state.global_step}/{self.total_steps} "
                      f"({state.global_step/self.total_steps*100:.1f}%) - "
                      f"Loss: {loss} - LR: {lr} - "
                      f"ETA: {eta/60:.1f} minutes")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[ProgressCallback(total_steps)]
    )
    
    print("\n" + "="*60)
    print("Starting IMPROVED training with better data formatting...")
    print(f"Total training steps: {total_steps}")
    print(f"Model: {config.model_name}")
    print(f"Training examples: {len(train_dataset)}")
    print("="*60)
    
    # Train
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    print(f"\n✅ Training completed in {training_time/60:.1f} minutes!")
    
    # Save the model
    print("\nSaving model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Save training info
    training_info = {
        "base_model": config.model_name,
        "model_size": "124M parameters",
        "training_time_minutes": training_time/60,
        "examples_trained": len(train_dataset),
        "total_steps": total_steps,
        "final_loss": trainer.state.log_history[-1].get('loss', 'N/A'),
        "device": str(device),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(config.output_dir / "training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    
    return trainer


def test_generation(model, tokenizer, config):
    """Test the fine-tuned model."""
    print("\n" + "="*60)
    print("Testing improved model on bio completions...")
    print("="*60)
    
    test_prompts = [
        "We are a couple who",
        "Looking for friends to",
        "I enjoy outdoor activities and",
        "My hobbies include",
        "We are interested in meeting",
        "I am looking for someone who",
        "Fun couple seeking",
        "Professional couple looking for"
    ]
    
    model.eval()
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated[len(prompt):].strip()
        
        print(f"\nPrompt: '{prompt}'")
        print(f"Completion: '{completion}'")


def main():
    """Main training function."""
    print("=== Improved Bio Autocomplete Training ===")
    print(f"Device: {device}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    config = ImprovedBioTrainingConfig()
    
    try:
        # Load model
        model, tokenizer = load_and_prepare_model(config)
        
        # Prepare data
        train_dataset, val_dataset = prepare_datasets(config, tokenizer)
        
        # Train
        trainer = train_model(model, tokenizer, train_dataset, val_dataset, config)
        
        # Test
        test_generation(model, tokenizer, config)
        
        print("\n✅ Improved training complete!")
        print(f"Model saved to: {config.output_dir}")
        print("\nNext steps:")
        print("1. Test the model more thoroughly")
        print("2. If results are good, train on full dataset")
        print("3. Convert to production format")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n🎯 This training uses IMPROVED data formatting!")
    print("Expected: Better quality completions")
    print("Model: GPT2 (124M params)")
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    main()