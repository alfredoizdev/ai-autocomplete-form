#!/bin/bash

# Start Ultrathink in TRAINED MODEL MODE
# This mode uses your fine-tuned MLX model for fast, specialized completions

echo "🤖 Starting TRAINED MODEL MODE"
echo "==========================================="
echo ""
echo "Trained model mode uses:"
echo "  ✓ Fine-tuned Llama-3.2-3B model"
echo "  ✓ Trained on high-quality sentence data"
echo "  ✓ Natural, grammatically correct completions"
echo "  ✓ Fast response times (50-100ms)"
echo ""

# Set the mode in .env.local
sed -i '' 's/AUTOCOMPLETE_MODE=.*/AUTOCOMPLETE_MODE=trained/' .env.local
echo "✅ Set AUTOCOMPLETE_MODE=trained in .env.local"

# Check if MLX server is already running
if lsof -Pi :8003 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  MLX Server is already running on port 8003"
    echo ""
else
    echo "Starting MLX Model Server on port 8003..."
    cd python && source venv/bin/activate && cd mlx_server && nohup python mlx_model_server.py > mlx_server.log 2>&1 &
    cd ../..
    
    # Wait for server to start
    sleep 5
    
    if lsof -Pi :8003 -sTCP:LISTEN -t >/dev/null ; then
        echo "✅ MLX Server started successfully"
    else
        echo "❌ Failed to start MLX Server"
        echo "Check python/mlx_server/mlx_server.log for errors"
        echo ""
        echo "Make sure you have trained the model first!"
        echo "Model should be at: models/bio-sentence-llama3-lora/"
        echo ""
        echo "To train the model, run: ./start_sentence_training.sh"
        exit 1
    fi
fi

echo ""
echo "🎉 Trained model mode is ready!"
echo ""
echo "Services running:"
echo "  - MLX Server: http://localhost:8003"
echo "  - Model: Fine-tuned Llama-3.2-3B (bio-sentence-llama3-lora)"
echo ""
echo "Next step:"
echo "  Open a new terminal and run: npm run dev"
echo ""
echo "To stop trained model mode:"
echo "  Press Ctrl+C here"
echo "  Kill MLX Server: lsof -ti:8003 | xargs kill"
echo ""

# Keep script running to show it's active
echo "Press Ctrl+C to stop monitoring..."
tail -f python/mlx_server/mlx_server.log