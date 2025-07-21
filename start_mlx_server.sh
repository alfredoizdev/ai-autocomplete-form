#!/bin/bash

echo "🚀 Starting MLX Bio Autocomplete API Server"
echo "=========================================="
echo "This will serve your fine-tuned MLX model"
echo ""

# Check if Python 3.11 is available
if ! command -v /opt/homebrew/bin/python3.11 &> /dev/null; then
    echo "❌ Python 3.11 not found at /opt/homebrew/bin/python3.11"
    echo "Using default python3 instead..."
    PYTHON_CMD="python3"
else
    PYTHON_CMD="/opt/homebrew/bin/python3.11"
    echo "✅ Using Python 3.11"
fi

# Navigate to the API directory
cd "$(dirname "$0")/python/api" || exit

# Start the MLX API server
echo ""
echo "🤖 Loading MLX model and starting server..."
echo "Server will be available at: http://localhost:8003"
echo "API docs at: http://localhost:8003/docs"
echo ""

$PYTHON_CMD mlx_api_server.py