#!/bin/bash

# Start the Bio Autocomplete API Server

echo "Starting Bio Autocomplete API Server..."
echo "======================================="
echo ""
echo "This server provides hybrid autocomplete functionality for bio text."
echo "It combines vector search with LLM generation for better completions."
echo ""
echo "API will be available at: http://localhost:8001"
echo "API documentation at: http://localhost:8001/docs"
echo ""
echo "Make sure:"
echo "1. Ollama is running (http://localhost:11434)"
echo "2. Next.js dev server is running (npm run dev)"
echo ""
echo "Press Ctrl+C to stop the server"
echo "======================================="
echo ""

# Navigate to python directory and start the server
cd python && python3 api/api_server.py