#!/bin/bash
# Setup script for vector database

echo "Setting up Python environment for vector database..."

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete! Virtual environment is activated."
echo "To run the ChromaDB setup, use: python vector_db/setup_chromadb.py"