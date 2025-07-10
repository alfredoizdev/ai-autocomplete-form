#!/bin/bash
# Run the FastAPI server

echo "Starting Bio Autocomplete API server..."

# Activate virtual environment
source venv/bin/activate

# Run the server
python api/api_server.py