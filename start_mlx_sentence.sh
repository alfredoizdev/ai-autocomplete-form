#!/bin/bash

echo "🚀 Starting MLX Bio Autocomplete - SENTENCE ONLY Mode"
echo "===================================================="
echo "✅ Strict single sentence completions"
echo "✅ Maximum 15 words"
echo "✅ Aggressive truncation"
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

# Start the sentence-only MLX API server
echo ""
echo "🤖 Starting sentence-only MLX server..."
nohup $PYTHON_CMD python/api/mlx_api_sentence.py > mlx_sentence.log 2>&1 &
MLX_PID=$!

echo "✅ MLX server started with PID: $MLX_PID"
echo "📝 Logs: mlx_sentence.log"
echo ""
echo "🌐 API: http://localhost:8003"
echo "📚 Docs: http://localhost:8003/docs"
echo ""
echo "Features:"
echo "  🎯 ONE sentence completion only"
echo "  🎯 Max 15 words per completion"
echo "  🎯 Stops at first period/punctuation"
echo "  🎯 No multi-paragraph generation"
echo ""
echo "To stop: kill $MLX_PID"
echo "To view logs: tail -f mlx_sentence.log"