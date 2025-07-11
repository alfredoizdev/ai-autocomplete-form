# API Server Guide

## Overview
The Bio Autocomplete API server provides hybrid autocomplete functionality that combines vector search with LLM generation for better bio text completions.

## Quick Start

### 1. Start the API Server
```bash
./start_api_server.sh
```

Or manually:
```bash
cd python
python3 api/api_server.py
```

The server will run on **http://localhost:8001**

### 2. Verify It's Running
```bash
curl http://localhost:8001/
```

Should return:
```json
{
    "status": "ok",
    "service": "Bio Autocomplete API",
    "vector_search_ready": true
}
```

## Required Services

1. **API Server** (port 8001) - This server
2. **Ollama** (port 11434) - For LLM generation
3. **Next.js** (port 3000) - Your frontend app

## API Endpoints

### Health Check
- **GET** `/`
- Returns server status

### Hybrid Autocomplete
- **POST** `/api/autocomplete/hybrid`
- Body: `{"prompt": "your text here"}`
- Returns:
  - `exact_matches`: Vector search results
  - `llm_completions`: AI-generated completions
  - `combined_suggestions`: Best of both

### Stats
- **GET** `/api/stats`
- Returns vector database statistics

## Troubleshooting

### ECONNREFUSED Error
This means the API server isn't running. Start it with:
```bash
./start_api_server.sh
```

### Port Already in Use
Kill the existing process:
```bash
lsof -i :8001
kill -9 <PID>
```

### Missing Dependencies
Install required packages:
```bash
cd python
pip3 install -r requirements.txt
```

## Development

API documentation is available at:
**http://localhost:8001/docs**

The server uses:
- FastAPI for the web framework
- ChromaDB for vector search
- Ollama for LLM generation