#!/bin/bash

echo "🚀 Starting Improved MLX Bio Autocomplete API Server"
echo "===================================================="
echo "Features: Minimum word enforcement, better prompting"
echo ""

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

# Start the improved MLX API server in background
echo ""
echo "🤖 Starting improved MLX server..."
nohup $PYTHON_CMD python/api/mlx_api_improved.py > mlx_server_improved.log 2>&1 &
MLX_PID=$!

echo "✅ MLX server started with PID: $MLX_PID"
echo "📝 Logs: mlx_server_improved.log"
echo ""
echo "🌐 API: http://localhost:8003"
echo "📚 Docs: http://localhost:8003/docs"
echo ""
echo "Features:"
echo "  ✅ Minimum 8 words per completion"
echo "  ✅ Multiple prompt strategies"
echo "  ✅ Repetition penalty enabled"
echo "  ✅ Concurrent generation for variety"
echo ""
echo "To stop: kill $MLX_PID"
echo "To view logs: tail -f mlx_server_improved.log"