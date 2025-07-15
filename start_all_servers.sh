#!/bin/bash

# Script to start all required servers for the AI Bio Generator

echo "🚀 Starting all servers for AI Bio Generator..."
echo ""

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠️  Ollama is not running. Please start it with: ollama serve"
    echo "   Also ensure you have the model: ollama pull gemma3:12b"
    echo ""
else
    echo "✅ Ollama is running"
fi

# Function to start a server in the background
start_server() {
    local name=$1
    local command=$2
    local port=$3
    local log_file=$4
    
    # Check if server is already running on the port
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  $name is already running on port $port"
    else
        echo "Starting $name on port $port..."
        nohup $command > $log_file 2>&1 &
        
        # Wait a bit and check if it started
        sleep 3
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
            echo "✅ $name started successfully"
        else
            echo "❌ Failed to start $name - check $log_file for errors"
        fi
    fi
}

# Navigate to python directory
cd python

# Start API Server (port 8001)
start_server "API Server" "python3 api/api_server.py" 8001 "api_server.log"

# Start Trained Model Server (port 8002)
start_server "Trained Model Server" "python3 -m uvicorn api.trained_model_server:app --port 8002" 8002 "trained_model_server.log"

# Navigate back
cd ..

echo ""
echo "🎉 All servers should be running!"
echo ""
echo "Services:"
echo "  - Ollama: http://localhost:11434"
echo "  - API Server: http://localhost:8001 (docs at /docs)"
echo "  - Trained Model: http://localhost:8002 (docs at /docs)"
echo "  - Next.js App: http://localhost:3000 (run 'npm run dev' to start)"
echo ""
echo "To stop servers:"
echo "  - Kill API Server: lsof -ti:8001 | xargs kill"
echo "  - Kill Trained Model: lsof -ti:8002 | xargs kill"
echo ""
echo "Check logs:"
echo "  - API Server: tail -f python/api_server.log"
echo "  - Trained Model: tail -f python/trained_model_server.log"