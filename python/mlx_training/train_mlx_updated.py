#!/usr/bin/env python3
"""
Updated MLX Training Script for Bio Autocomplete Fine-tuning.
Designed for MacBook Pro M1 Max with 32GB RAM.
Uses latest MLX-LM features (2025).
"""

import os
import json
import yaml
import time
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import sys

# Ensure we're using the right Python environment
print(f"Python: {sys.executable}")
print(f"Working directory: {os.getcwd()}")


class MLXTrainingPipeline:
    """Complete MLX training pipeline for bio autocomplete."""
    
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        self.model_name = model_name
        self.base_dir = Path(__file__).parent
        self.adapter_dir = self.base_dir / "adapters" / "bio_mistral_lora"
        self.data_dir = self.base_dir / "bio_dataset"
        self.config_path = self.base_dir / "training_config.yaml"
        
    def check_dependencies(self):
        """Check if all required packages are installed."""
        print("🔍 Checking dependencies...")
        
        required_packages = ["mlx", "mlx_lm", "transformers", "datasets"]
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                print(f"✅ {package} is installed")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package} is NOT installed")
        
        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False
            
        # Check MLX version
        try:
            import mlx
            print(f"✅ MLX version: {mlx.__version__}")
        except:
            pass
            
        return True
    
    def prepare_data(self):
        """Convert existing bio dataset to MLX format."""
        print("\n📊 Preparing training data...")
        
        # Check if data already exists
        train_file = self.data_dir / "train.jsonl"
        valid_file = self.data_dir / "validation.jsonl"
        
        if train_file.exists() and valid_file.exists():
            print("✅ Training data already prepared")
            return True
        
        try:
            from datasets import load_from_disk
            
            # Load existing datasets
            train_dataset = load_from_disk(str(self.data_dir / "train"))
            val_dataset = load_from_disk(str(self.data_dir / "validation"))
            
            print(f"Found {len(train_dataset)} training examples")
            print(f"Found {len(val_dataset)} validation examples")
            
            # Convert to JSONL format for MLX
            def save_as_jsonl(dataset, output_path):
                with open(output_path, 'w') as f:
                    for item in dataset:
                        # Format for instruction tuning
                        formatted = {
                            "text": f"[INST] Complete the following bio: {item['prompt']} [/INST] {item['completion']}"
                        }
                        f.write(json.dumps(formatted) + '\n')
            
            save_as_jsonl(train_dataset, train_file)
            save_as_jsonl(val_dataset, valid_file)
            
            print("✅ Data converted to MLX format")
            return True
            
        except Exception as e:
            print(f"❌ Error preparing data: {e}")
            return False
    
    def create_training_config(self):
        """Create optimized training configuration for M1 Max 32GB."""
        print("\n⚙️  Creating training configuration...")
        
        config = {
            "model": self.model_name,
            "train": True,
            "data": str(self.data_dir),
            "seed": 42,
            
            # LoRA configuration optimized for 32GB RAM
            "lora_layers": 16,  # Can increase to 32 if memory allows
            "lora_parameters": {
                "rank": 8,  # Start conservative, can increase to 16
                "alpha": 16,
                "dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
            },
            
            # Training hyperparameters
            "batch_size": 2,  # Small batch for safety
            "gradient_accumulation_steps": 4,  # Effective batch = 8
            "learning_rate": 2e-4,
            "warmup_steps": 100,
            "num_epochs": 1,  # Start with 1, can increase
            
            # Memory optimization
            "max_seq_length": 512,  # Bio texts are typically short
            "gradient_checkpointing": True,
            
            # Evaluation and checkpointing
            "eval_steps": 100,
            "save_every": 250,
            "test_every": 500,
            
            # Output configuration
            "adapter_path": str(self.adapter_dir),
            
            # Logging
            "wandb": "bio-autocomplete-mlx",  # Optional W&B logging
        }
        
        # Save configuration
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Configuration saved to {self.config_path}")
        return True
    
    def download_model(self):
        """Download and convert model if needed."""
        print(f"\n📥 Checking model: {self.model_name}")
        
        # Check if model already exists locally
        model_cache = Path.home() / ".cache" / "huggingface" / "hub"
        
        # For now, just provide instructions
        print("\nTo download the model, run:")
        print(f"python -m mlx_lm.convert --hf-path {self.model_name} -q")
        print("\nThis will download and quantize the model for MLX.")
        print("The 4-bit quantized version will use ~4GB of storage.")
        
        return True
    
    def run_training(self):
        """Execute the training process."""
        print("\n🚀 Starting training...")
        print("="*50)
        
        # Build training command
        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--config", str(self.config_path),
            "--train"
        ]
        
        print(f"Command: {' '.join(cmd)}")
        print("\n⏱️  Expected training time: 3-4 hours")
        print("💾 Expected memory usage: 20-25GB")
        print("\nMonitor progress in Activity Monitor")
        print("="*50)
        
        # Create training script for easy execution
        script_path = self.base_dir / "run_training.sh"
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# MLX Training Script\n")
            f.write(f"# Generated on {datetime.now()}\n\n")
            f.write(f"cd {self.base_dir}\n")
            f.write(f"{' '.join(cmd)}\n")
        
        os.chmod(script_path, 0o755)
        print(f"\n✅ Training script created: {script_path}")
        print("Run with: ./run_training.sh")
        
        return True
    
    def create_inference_script(self):
        """Create a script for testing the trained model."""
        print("\n📝 Creating inference script...")
        
        inference_code = '''#!/usr/bin/env python3
"""Test the fine-tuned bio autocomplete model."""

from mlx_lm import load, generate
import sys

def test_model(prompt="We are a couple who"):
    """Test the fine-tuned model with a prompt."""
    
    model_path = "mistralai/Mistral-7B-Instruct-v0.2"
    adapter_path = "./adapters/bio_mistral_lora"
    
    print(f"Loading model with adapter...")
    model, tokenizer = load(model_path, adapter_path=adapter_path)
    
    print(f"\\nPrompt: {prompt}")
    print("Generating completion...")
    
    response = generate(
        model, 
        tokenizer, 
        prompt=f"[INST] Complete the following bio: {prompt} [/INST]",
        max_tokens=100,
        temperature=0.7
    )
    
    print(f"\\nResponse: {response}")

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "We are a couple who"
    test_model(prompt)
'''
        
        script_path = self.base_dir / "test_finetuned_model.py"
        with open(script_path, 'w') as f:
            f.write(inference_code)
        
        os.chmod(script_path, 0o755)
        print(f"✅ Inference script created: {script_path}")
        
        return True
    
    def run_pipeline(self):
        """Run the complete training pipeline."""
        print("🎯 MLX Training Pipeline for Bio Autocomplete")
        print(f"Model: {self.model_name}")
        print(f"Hardware: MacBook Pro M1 Max 32GB RAM")
        print("="*50)
        
        steps = [
            ("Check dependencies", self.check_dependencies),
            ("Prepare data", self.prepare_data),
            ("Create config", self.create_training_config),
            ("Setup model", self.download_model),
            ("Setup training", self.run_training),
            ("Create inference", self.create_inference_script),
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'='*50}")
            print(f"Step: {step_name}")
            print('='*50)
            
            if not step_func():
                print(f"\n❌ Pipeline failed at: {step_name}")
                return False
        
        print("\n✅ Pipeline setup complete!")
        print("\nNext steps:")
        print("1. Ensure Hugging Face CLI is logged in: huggingface-cli login")
        print("2. Download model: python -m mlx_lm.convert --hf-path mistralai/Mistral-7B-Instruct-v0.2 -q")
        print("3. Run training: ./run_training.sh")
        print("4. Test model: python test_finetuned_model.py")
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MLX Training Pipeline")
    parser.add_argument(
        "--model", 
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Model to fine-tune"
    )
    
    args = parser.parse_args()
    
    pipeline = MLXTrainingPipeline(model_name=args.model)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()