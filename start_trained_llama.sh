#!/bin/bash

echo "=================================="
echo "Starting MLX Server with Trained Llama 3.2 Model"
echo "=================================="
echo ""
echo "Model: Llama-3.2-3B-Instruct with LoRA fine-tuning"
echo "Training: 1000 iterations on bio completion dataset"
echo "Port: 8003"
echo ""

# Navigate to project directory
cd "$(dirname "$0")"

# Check if the trained model exists
if [ ! -f "models/bio-sentence-llama3-lora-continued/adapters.safetensors" ]; then
    echo "❌ Error: Trained model not found at models/bio-sentence-llama3-lora-continued/"
    echo "Please run the training script first: ./start_sentence_training.sh"
    exit 1
fi

echo "✅ Found trained model at models/bio-sentence-llama3-lora-continued/"
echo ""

# Start the MLX server
echo "Starting MLX server on port 8003..."
echo "Server will be available at http://localhost:8003"
echo "API endpoint: http://localhost:8003/api/autocomplete/mlx"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the server
python3 python/mlx_server/mlx_model_server.py