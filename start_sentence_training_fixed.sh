#!/bin/bash

echo "=================================="
echo "SENTENCE COMPLETION MODEL TRAINING"
echo "=================================="
echo ""
echo "Training with:"
echo "- High-quality sentence data"
echo "- Llama 3.2 3B model (already cached)"
echo "- Natural prompt-completion pairs"
echo ""

# Set the base directory
BASE_DIR="/Users/simonlacey/Documents/GitHub/llms/ai-train-llm"
cd "$BASE_DIR"

echo "Working directory: $(pwd)"
echo ""

# Check that files exist
echo "Checking training files..."
ls -la python/sentence_training/mlx/*.jsonl
echo ""

# Update config to use absolute paths
cat > python/sentence_training/mlx/config_absolute.yaml << EOF
# MLX LoRA Training Configuration for Bio Sentence Completion
# High-quality sentence-based training for natural completions

# Model configuration
model: "mlx-community/Llama-3.2-3B-Instruct-4bit"
tokenizer_config: {}
trust_remote_code: true

# Training data paths (absolute)
train: "$BASE_DIR/python/sentence_training/mlx/train.jsonl"
valid: "$BASE_DIR/python/sentence_training/mlx/validation.jsonl" 
test: "$BASE_DIR/python/sentence_training/mlx/test.jsonl"

# LoRA adapter configuration
adapter_path: "$BASE_DIR/models/bio-sentence-llama3-lora"
resume_adapter_path: null

# Training hyperparameters
seed: 42
num_layers: 16  # Number of layers to apply LoRA to
batch_size: 4
iters: 1000  # Reduced iterations since we have focused, high-quality data
val_batches: 25
learning_rate: 1e-4  # Conservative learning rate
warmup_steps: 100
save_every: 200
test_every: 200

# LoRA specific parameters
lora_rank: 16  # Higher rank for more expressiveness
lora_alpha: 16
lora_dropout: 0.1
lora_keys: ["self_attn.q_proj", "self_attn.v_proj", "self_attn.k_proj", "self_attn.o_proj"]

# Data processing
max_seq_length: 256  # Shorter sequences for sentence completion
train_on_completions: true  # Only train on the completion part

# Model saving
save_safetensors: true

# Training optimizations
grad_checkpoint: false  # Disable for faster training with smaller model
use_dora: false  # Disable DoRA for stability
EOF

echo "Starting training with absolute paths..."
echo ""

python3 -m mlx_lm lora --config python/sentence_training/mlx/config_absolute.yaml

echo ""
echo "Training complete!"
echo ""
echo "Next steps:"
echo "1. Update mlx_model_server.py to use 'models/bio-sentence-llama3-lora'"
echo "2. Restart the MLX server"
echo "3. Test the improved completions"