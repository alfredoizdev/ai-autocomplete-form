"""
Efficient training script for bio autocomplete using smaller models.
Optimized for fast training on M1 Mac with 32GB RAM.
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


class EfficientBioTrainingConfig:
    """Efficient training configuration for bio autocomplete."""
    
    def __init__(self):
        # Use DistilGPT2 - only 82M parameters, perfect for autocomplete
        self.model_name = "distilgpt2"  
        self.output_dir = Path(__file__).parent / "bio_distilgpt2_finetuned"
        self.data_path = Path(__file__).parent / "bio_dataset"
        
        # Quality-focused settings
        self.batch_size = 4  # Smaller batch for better gradient quality
        self.gradient_accumulation_steps = 4  # Effective batch size of 16
        self.learning_rate = 3e-4  # Moderate LR for stable training
        self.num_train_epochs = 3  # More epochs for better quality
        self.max_seq_length = 96  # Longer context for better understanding
        self.warmup_steps = 50
        self.logging_steps = 50
        self.save_steps = 1000
        self.eval_steps = 500
        
        # LoRA configuration for even faster training
        self.use_lora = True
        self.lora_r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.1
        self.lora_target_modules = ["c_attn", "c_proj"]  # GPT2 attention layers
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)


def load_and_prepare_model(config):
    """Load small model with LoRA for fast training."""
    print(f"\nLoading {config.model_name} - a lightweight model perfect for autocomplete...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model - DistilGPT2 is much smaller and faster
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,  # MPS works better with float32
        device_map={"": device}
    )
    
    # Get model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params/1e6:.1f}M parameters (30x smaller than Phi-2!)")
    
    if config.use_lora:
        print("Applying LoRA for ultra-fast training...")
        
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


def prepare_datasets(config, tokenizer, sample_size=None):
    """Prepare training data with optional sampling for quick tests."""
    print("\nLoading datasets...")
    train_dataset = load_from_disk(str(config.data_path / "train"))
    val_dataset = load_from_disk(str(config.data_path / "validation"))
    
    print(f"Full dataset: {len(train_dataset)} training examples")
    
    # Option to use a subset for even faster training
    if sample_size:
        train_dataset = train_dataset.select(range(min(sample_size, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(sample_size//10, len(val_dataset))))
        print(f"Using subset: {len(train_dataset)} training examples")
    
    def tokenize_function(examples):
        # Simple format for autocomplete
        texts = []
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            # Format: prompt -> completion
            text = f"{prompt} {completion}"
            texts.append(text)
        
        return tokenizer(
            texts,
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length"
        )
    
    # Tokenize datasets with multiprocessing
    print("Tokenizing datasets (this should be fast)...")
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
    """Train the model with progress tracking."""
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Calculate total steps
    total_steps = len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps) * config.num_train_epochs
    
    # Training arguments optimized for speed
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        overwrite_output_dir=True,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,  # Larger eval batch
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
        save_total_limit=1,  # Save space
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Important for MPS
        optim="adamw_torch",
        prediction_loss_only=True,  # Faster evaluation
        gradient_checkpointing=False,  # Not needed for small model
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
    print("Starting QUALITY-FOCUSED training with bio autocomplete data...")
    print(f"Total training steps: {total_steps}")
    print(f"Training epochs: {config.num_train_epochs}")
    print(f"Estimated time: {total_steps * 0.8 / 60:.1f} minutes")
    print("="*60)
    
    # Train with timing
    start_time = time.time()
    
    # Add progress callback
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
                print(f"\nStep {state.global_step}/{self.total_steps} "
                      f"({state.global_step/self.total_steps*100:.1f}%) - "
                      f"Loss: {loss} - "
                      f"Speed: {1/steps_per_sec:.2f}s/step - "
                      f"ETA: {eta/60:.1f} minutes")
    
    trainer.add_callback(ProgressCallback(total_steps))
    
    # Train
    trainer.train()
    training_time = time.time() - start_time
    
    print(f"\n✅ Training completed in {training_time/60:.1f} minutes!")
    print(f"Average speed: {total_steps/training_time:.1f} steps/second")
    
    # Save the model
    print("\nSaving model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Save training info
    training_info = {
        "base_model": config.model_name,
        "model_size": "82M parameters",
        "training_time_minutes": training_time/60,
        "examples_trained": len(train_dataset),
        "total_steps": total_steps,
        "average_speed_steps_per_sec": total_steps/training_time,
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
                max_new_tokens=15,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated[len(prompt):].strip()
        
        print(f"\nPrompt: '{prompt}'")
        print(f"Completion: '{completion}'")


def main():
    """Main training function."""
    print("=== Quality-Focused Bio Autocomplete Training ===")
    print(f"Device: {device}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    config = EfficientBioTrainingConfig()
    
    try:
        # Load model
        model, tokenizer = load_and_prepare_model(config)
        
        # Prepare data - use full dataset for production
        # Use sample_size=5000 for a quick test run
        train_dataset, val_dataset = prepare_datasets(config, tokenizer, sample_size=None)
        
        # Train
        trainer = train_model(model, tokenizer, train_dataset, val_dataset, config)
        
        # Test
        test_generation(model, tokenizer, config)
        
        print("\n✅ Training complete!")
        print(f"Model saved to: {config.output_dir}")
        print("\nNext steps:")
        print("1. Model is ready to use with transformers library")
        print("2. Can be converted to ONNX for faster inference")
        print("3. Can be integrated with your autocomplete API")
        
        # Quick size check
        model_size = sum(os.path.getsize(config.output_dir / f) 
                        for f in os.listdir(config.output_dir) 
                        if f.endswith(('.bin', '.safetensors'))) / 1e6
        print(f"\nModel size on disk: {model_size:.1f} MB")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n🎯 This training script is optimized for QUALITY!")
    print("Expected training time: 90-120 minutes")
    print("Model: DistilGPT2 (82M params)")
    print("Training: 3 epochs on 35,357 examples")
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    main()