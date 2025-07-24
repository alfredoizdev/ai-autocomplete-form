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
    echo "⚠️  Ollama is not running. Starting Ollama..."
    # Start Ollama in the background
    ollama serve > /dev/null 2>&1 &
    
    # Give Ollama time to start
    sleep 5
    
    # Check if it started successfully
    if pgrep -x "ollama" > /dev/null; then
        echo "✅ Ollama started successfully"
    else
        echo "❌ Failed to start Ollama"
        echo "Please start it manually with: ollama serve"
        exit 1
    fi
else
    echo "✅ Ollama is already running"
fi

# Check if gemma3:12b model is available
echo "Checking for gemma3:12b model..."
if ollama list | grep -q "gemma3:12b"; then
    echo "✅ gemma3:12b model is available"
else
    echo "⚠️  gemma3:12b model not found. Pulling model..."
    echo "This may take a few minutes for the first download..."
    
    if ollama pull gemma3:12b; then
        echo "✅ Successfully pulled gemma3:12b model"
    else
        echo "❌ Failed to pull gemma3:12b model"
        echo "Please run manually: ollama pull gemma3:12b"
        exit 1
    fi
fi

# Check if API server is already running
if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  API Server is already running on port 8001"
    echo ""
else
    echo "Starting API Server on port 8001..."
    cd python && source venv/bin/activate && nohup python -m uvicorn api.api_server:app --host 0.0.0.0 --port 8001 > api_server.log 2>&1 &
    cd ..
    
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