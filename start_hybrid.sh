#!/bin/bash

# Start Ultrathink in HYBRID MODE
# This mode uses vector search + AI generation for the best results

echo "🚀 Starting in HYBRID MODE"
echo "===================================="
echo ""
echo "Hybrid mode combines:"
echo "  ✓ Vector search (finds similar bios)"
echo "  ✓ AI generation (creates new completions)"
echo ""

# Set the mode in .env.local
sed -i '' 's/AUTOCOMPLETE_MODE=.*/AUTOCOMPLETE_MODE=hybrid/' .env.local
echo "✅ Set AUTOCOMPLETE_MODE=hybrid in .env.local"

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠️  Ollama is not running!"
    echo ""
    echo "Please start Ollama first:"
    echo "  ollama serve"
    echo ""
    echo "Then make sure you have the model:"
    echo "  ollama pull gemma3:12b"
    echo ""
    exit 1
else
    echo "✅ Ollama is running"
fi

# Check if API server is already running
if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  API Server is already running on port 8001"
    echo ""
else
    echo "Starting API Server on port 8001..."
    cd python/api && source ../venv/bin/activate && nohup python api_server.py > ../api_server.log 2>&1 &
    cd ../..
    
    # Wait for server to start
    sleep 3
    
    if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null ; then
        echo "✅ API Server started successfully"
    else
        echo "❌ Failed to start API Server"
        echo "Check python/api_server.log for errors"
        exit 1
    fi
fi

echo ""
echo "🎉 Hybrid mode is ready!"
echo ""
echo "Services running:"
echo "  - Ollama: http://localhost:11434"
echo "  - API Server: http://localhost:8001 (docs at /docs)"
echo ""
echo "Next step:"
echo "  Open a new terminal and run: npm run dev"
echo ""
echo "To stop hybrid mode:"
echo "  Press Ctrl+C here"
echo "  Kill API Server: lsof -ti:8001 | xargs kill"
echo ""

# Keep script running to show it's active
echo "Press Ctrl+C to stop monitoring..."
tail -f python/api_server.log