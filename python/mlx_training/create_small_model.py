"""
Create a small language model for bio autocomplete training.
This uses a much smaller model that can be trained on M1 Mac with 32GB RAM.
"""

import json
from pathlib import Path
from transformers import (
    GPT2Config, 
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    AutoTokenizer
)
import torch

def create_small_model():
    """Create a small GPT2-style model for training."""
    
    output_dir = Path(__file__).parent / "small_model"
    output_dir.mkdir(exist_ok=True)
    
    print("Creating small model for bio autocomplete...")
    
    # Configuration for a very small model
    config = GPT2Config(
        vocab_size=50257,  # Standard GPT2 vocab size
        n_positions=256,   # Max sequence length
        n_embd=512,       # Hidden size (small)
        n_layer=6,        # Only 6 layers
        n_head=8,         # 8 attention heads
        n_inner=2048,     # FFN size
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
    )
    
    # Create model
    print("Initializing model...")
    model = GPT2LMHeadModel(config)
    
    # Calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    print(f"Model created!")
    print(f"Total parameters: {total_params:,}")
    print(f"Estimated size: {model_size_mb:.1f} MB")
    
    # Save model
    print(f"\nSaving model to {output_dir}")
    model.save_pretrained(output_dir)
    
    # Use GPT2 tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output_dir)
    
    # Save model info
    model_info = {
        "model_type": "gpt2-small-custom",
        "total_params": total_params,
        "size_mb": model_size_mb,
        "config": config.to_dict(),
        "purpose": "Bio autocomplete fine-tuning",
        "optimized_for": "M1 Mac 32GB RAM"
    }
    
    with open(output_dir / "model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\n✅ Small model created successfully!")
    print(f"Location: {output_dir}")
    
    return output_dir


def test_model(model_dir):
    """Test the created model."""
    print("\n" + "="*50)
    print("Testing model...")
    print("="*50)
    
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    
    # Test generation
    text = "We are a couple"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=20,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: '{text}'")
    print(f"Generated: '{generated}'")
    print("\nNote: This is untrained output. Training will improve quality.")


if __name__ == "__main__":
    print("=== Creating Small Model for Bio Autocomplete ===")
    print("\nThis creates a 128MB model that can be trained efficiently on M1 Mac.")
    
    model_dir = create_small_model()
    test_model(model_dir)
    
    print("\n🚀 Next step:")
    print(f"python mlx_training/train_small_model.py")