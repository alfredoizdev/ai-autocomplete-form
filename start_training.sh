#!/bin/bash

# Start MLX training for local LLM fine-tuning
# This script trains a LoRA adapter on the bio dataset

echo "🤖 Starting Local LLM Training with MLX"
echo "======================================="
echo ""

# Configuration
MODEL_NAME="mlx-community/Llama-3.2-1B-Instruct-4bit"
TRAINING_DATA_DIR="python/lookingfor_mlx"
OUTPUT_DIR="models/lookingfor-llama3-lora"
LEARNING_RATE=5e-5
BATCH_SIZE=4
NUM_ITERATIONS=1000
SAVE_EVERY=100

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Training data: $TRAINING_DATA_DIR"
echo "  Output: $OUTPUT_DIR"
echo ""

# Check if Python venv exists
if [ ! -d "python/venv" ]; then
    echo "❌ Python virtual environment not found!"
    echo "Please run: cd python && python -m venv venv && pip install -r requirements.txt"
    exit 1
fi

# Check if training data exists
if [ ! -f "$TRAINING_DATA_DIR/train.jsonl" ]; then
    echo "❌ Training data not found at $TRAINING_DATA_DIR!"
    echo ""
    echo "To prepare training data, run:"
    echo "  ./prepare_lookingfor_data.sh"
    exit 1
fi

# Activate virtual environment
echo "Activating Python environment..."
cd python
source venv/bin/activate

# Check if mlx_lm is installed
if ! python -c "import mlx_lm" 2>/dev/null; then
    echo "Installing MLX-LM..."
    pip install mlx-lm
fi

# Create output directory if it doesn't exist
mkdir -p ../$OUTPUT_DIR

echo ""
echo "Starting training..."
echo "==================="
echo ""

# Run MLX LoRA training
python -m mlx_lm.lora \
    --model $MODEL_NAME \
    --train \
    --data ../$TRAINING_DATA_DIR \
    --batch-size $BATCH_SIZE \
    --lora-layers 16 \
    --iters $NUM_ITERATIONS \
    --val-batches 25 \
    --learning-rate $LEARNING_RATE \
    --warmup 100 \
    --save-every $SAVE_EVERY \
    --adapter-path ../$OUTPUT_DIR

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo ""
    echo "Model saved to: $OUTPUT_DIR"
    echo ""
    echo "To use the trained model:"
    echo "  1. Update start_trained.sh to point to $OUTPUT_DIR"
    echo "  2. Run: ./start_trained.sh"
    echo "  3. In a new terminal: npm run dev"
else
    echo ""
    echo "❌ Training failed. Check the error messages above."
    exit 1
fi

# Deactivate virtual environment
deactivate
cd ..

echo ""
echo "🎉 Training complete!"
echo ""
echo "Training stats:"
echo "  - Iterations: $NUM_ITERATIONS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Learning rate: $LEARNING_RATE"
echo "  - Checkpoints saved every: $SAVE_EVERY iterations"
echo ""