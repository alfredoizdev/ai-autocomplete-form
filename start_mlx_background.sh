#!/bin/bash

echo "🚀 Starting MLX Bio Autocomplete API Server in background"
echo "========================================================"

# Check if Python 3.11 is available
if ! command -v /opt/homebrew/bin/python3.11 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="/opt/homebrew/bin/python3.11"
fi

# Navigate to the project directory
cd "$(dirname "$0")" || exit

# Kill any existing MLX server on port 8003
echo "🔍 Checking for existing MLX server..."
lsof -ti:8003 | xargs kill -9 2>/dev/null && echo "✅ Stopped existing MLX server" || echo "✅ No existing MLX server found"

# Start the MLX API server in background
echo ""
echo "🤖 Starting MLX server in background..."
nohup $PYTHON_CMD python/api/mlx_api_server.py > mlx_server.log 2>&1 &
MLX_PID=$!

echo "✅ MLX server started with PID: $MLX_PID"
echo "📝 Logs are being written to: mlx_server.log"
echo ""
echo "🌐 MLX API available at: http://localhost:8003"
echo "📚 API docs at: http://localhost:8003/docs"
echo ""
echo "To stop the server: kill $MLX_PID"
echo "To view logs: tail -f mlx_server.log"