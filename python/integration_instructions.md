# Python Services Integration Instructions

## Overview

The Python backend provides two main services for the bio autocomplete application:

1. **Hybrid Autocomplete API** (Port 8001) - Combines vector search with LLM generation
2. **MLX Model Server** (Port 8003) - Serves fine-tuned Llama models

## Quick Start

### Option 1: Hybrid Mode (Recommended for Quality)

```bash
# Start the hybrid autocomplete service
./start_hybrid.sh
```

This starts:
- FastAPI server on port 8001
- ChromaDB vector search
- Ollama integration for LLM generation
- Combined suggestions for best quality

### Option 2: Trained Model Mode (Recommended for Speed)

```bash
# Start the MLX model server
./start_trained.sh
```

This starts:
- MLX server on port 8003
- Fine-tuned Llama 3.2 models
- Fast inference on Apple Silicon
- Grammar-filtered high-quality completions

## Service Architecture

### Hybrid API Server (api_server.py)

```
Port: 8001
Endpoints:
- GET /                               # Health check
- POST /api/autocomplete              # Legacy vector-only endpoint
- POST /api/autocomplete/hybrid       # Hybrid autocomplete (main endpoint)
- GET /api/vector/stats               # Vector DB statistics
```

#### Request Format:
```json
{
  "prompt": "Looking for fun loving people"
}
```

#### Response Format:
```json
{
  "combined_suggestions": ["who enjoy life and good times"],
  "exact_matches": [],
  "llm_completions": ["who enjoy life and good times"],
  "context_used": "couples looking for...",
  "elapsed_ms": 125.4
}
```

### MLX Model Server (mlx_model_server.py)

```
Port: 8003
Endpoints:
- GET /health                         # Health check
- POST /api/autocomplete/mlx          # Single completion
- POST /api/autocomplete/mlx/batch    # Batch completions
- GET /docs                           # API documentation
```

#### Request Format:
```json
{
  "prompt": "Looking for fun loving people",
  "max_tokens": 50,
  "temperature": 0.7,
  "stop": [".", "!", "?", "\n"]
}
```

#### Response Format:
```json
{
  "completion": "that we can have fun with in and out of the bedroom",
  "elapsed_ms": 125.4,
  "model_name": "Llama-3.2-3B-hq (LoRA)"
}
```

## Next.js Integration

The integration is handled automatically through:

### 1. Environment Configuration (.env.local)

```bash
# Set the autocomplete mode
AUTOCOMPLETE_MODE=hybrid  # or 'trained'
```

### 2. Server Action (actions/ai-text.ts)

The `askOllamaCompletationAction` function automatically routes based on mode:

```typescript
// Check which mode to use
const mode = process.env.AUTOCOMPLETE_MODE || 'hybrid';

if (mode === 'trained') {
  // Use MLX model server
  const response = await fetch('http://localhost:8003/api/autocomplete/mlx', ...);
} else if (mode === 'hybrid') {
  // Use hybrid API server
  const response = await fetch('http://localhost:8001/api/autocomplete/hybrid', ...);
}
```

## Service Management

### Starting Services

```bash
# Hybrid mode (vector search + AI)
./start_hybrid.sh

# Trained model mode (fine-tuned Llama)
./start_trained.sh
```

### Checking Service Status

```bash
# Check if services are running
curl http://localhost:8001/  # Hybrid API
curl http://localhost:8003/health  # MLX server

# Check logs
tail -f python/api_server.log
tail -f python/mlx_server/mlx_server.log
```

### Stopping Services

```bash
# Find and kill the process
ps aux | grep "api_server.py"
ps aux | grep "mlx_model_server.py"
kill <PID>
```

## Performance Comparison

### Hybrid Mode (Port 8001)
- **Response Time**: 100-150ms
- **Quality**: Excellent (context-aware + LLM creativity)
- **Features**: Vector similarity + AI generation
- **Best For**: Production use with quality focus

### Trained Mode (Port 8003)
- **Response Time**: 50-150ms (model dependent)
- **Quality**: Very good (grammar-filtered training)
- **Features**: Direct model inference
- **Best For**: Fast responses, offline capability

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   lsof -i :8001  # or :8003
   kill -9 <PID>
   ```

2. **ChromaDB Errors**
   ```bash
   rm -rf python/vector_db/chroma_db
   python python/vector_db/setup_chromadb_improved.py
   ```

3. **Model Not Loading**
   - Check model files exist in `models/` directory
   - Verify MLX is installed: `pip install mlx mlx-lm`

4. **Slow Performance**
   - First request loads model (5-10s)
   - Subsequent requests are fast
   - Use 1B model for lower memory usage

## Advanced Configuration

### Custom Prompting

Both services support custom system prompts through their respective configurations:

- Hybrid: Modify prompt in `api_server.py`
- MLX: Adjust generation parameters in requests

### Scaling Considerations

- Both services are single-instance by default
- For production, consider:
  - Load balancer for multiple instances
  - Redis cache for common completions
  - Model quantization for memory efficiency

## Development Workflow

1. **Choose your mode** based on requirements
2. **Start the appropriate service** with startup script
3. **Set AUTOCOMPLETE_MODE** in .env.local
4. **Run Next.js app** with `npm run dev`
5. **Monitor logs** for debugging

The system automatically handles:
- Service health checks
- Fallback mechanisms
- Error handling
- Response formatting