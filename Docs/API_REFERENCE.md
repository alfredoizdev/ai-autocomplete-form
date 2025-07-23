# API Reference

Technical documentation for the AI Bio Autocomplete backend services.

## 🌐 Available Servers

### 1. Hybrid API Server (Port 8001)
- **Purpose**: Vector search + AI generation
- **Docs**: http://localhost:8001/docs
- **Start**: `./start_hybrid.sh`

### 2. MLX Model Server (Port 8003)
- **Purpose**: Fine-tuned model inference
- **Docs**: Built-in health endpoints
- **Start**: `./start_trained.sh`

## 📡 API Endpoints

### Hybrid API Server

#### `GET /`
Health check endpoint.
```bash
curl http://localhost:8001/
```

#### `POST /api/autocomplete`
Vector search only (fast exact matches).
```json
{
  "prompt": "I am looking for"
}
```

#### `POST /api/autocomplete/hybrid`
**Main endpoint** - Vector search + AI generation.
```json
{
  "prompt": "We are a fun couple who"
}
```

Response:
```json
{
  "combined_suggestions": [
    "enjoys meeting new people for friendship and fun",
    "likes to explore new experiences together",
    "is looking for like-minded couples"
  ],
  "exact_matches": [...],
  "llm_completions": [...],
  "context_used": 5,
  "elapsed_ms": 145.23
}
```

#### `GET /api/stats`
Database statistics.
```json
{
  "total_bios": 5000,
  "db_path": "chroma_db",
  "last_updated": "2024-01-20"
}
```

### MLX Model Server

#### `GET /`
Health check with model status.
```json
{
  "status": "ok",
  "model_loaded": true,
  "adapter_loaded": true
}
```

#### `POST /api/autocomplete/mlx`
Generate completion using fine-tuned model.
```json
{
  "prompt": "Looking for couples who",
  "max_tokens": 50,
  "temperature": 0.7
}
```

Response:
```json
{
  "completion": "enjoy dinners, dancing, and good conversation",
  "elapsed_ms": 67.89,
  "model_name": "llama3.2-mlx-finetuned"
}
```

## 🔧 Frontend Integration

### Server Actions (`actions/ai-text.ts`)

The frontend automatically selects the right endpoint based on mode:

```typescript
// Mode detection
const mode = process.env.AUTOCOMPLETE_MODE || 'hybrid';

if (mode === 'trained') {
  // Calls MLX server on port 8003
  const response = await fetch('http://localhost:8003/api/autocomplete/mlx', ...);
} else {
  // Calls hybrid API on port 8001
  const response = await fetch('http://localhost:8001/api/autocomplete/hybrid', ...);
}
```

## ⚙️ Configuration

### Hybrid API Server

Edit `python/api/api_server.py`:
```python
# Vector search settings
NUM_SIMILAR_BIOS = 10          # Similar bios to retrieve
MIN_SUGGESTION_LENGTH = 8      # Minimum words per suggestion

# AI generation settings  
OLLAMA_MODEL = "gemma3:12b"   # Model to use
TEMPERATURES = [0.7, 0.9]     # Generation variety
MAX_RETRIES = 3               # Ollama retry attempts
```

### MLX Model Server

Edit `python/mlx_server/mlx_model_server.py`:
```python
# Model loading priority
1. models/bio-sentence-llama3-lora-continued/
2. models/bio-sentence-llama3-lora/
3. Base model fallback

# Generation defaults
DEFAULT_MAX_TOKENS = 50
DEFAULT_TEMPERATURE = 0.7
```

## 📊 Performance Tuning

### Caching
The frontend caches responses for 5 minutes:
```typescript
// In actions/ai-text-streaming.ts
const CACHE_DURATION = 5 * 60 * 1000;
```

### Debouncing
Prevent too many API calls:
```typescript
// In hooks/useFormAutocomplete.tsx
const DEBOUNCE_DELAY = 1500;  // Wait 1.5s after typing
const MIN_WORDS_FOR_SUGGESTION = 5;
```

### Connection Pooling
Both servers use connection pooling for database/model access.

## 🚨 Error Responses

All endpoints return consistent error format:
```json
{
  "detail": "Error message",
  "status_code": 500
}
```

Common status codes:
- `200` - Success
- `422` - Invalid request data
- `500` - Server error
- `503` - Service unavailable (model not loaded)

## 🔐 Security

### CORS Configuration
Both servers allow requests from:
- `http://localhost:3000` (development)
- Configure for production in server files

### Rate Limiting
Not implemented by default. For production:
```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/autocomplete/hybrid")
@limiter.limit("10/minute")
async def autocomplete(...):
```

## 🧪 Testing Endpoints

### Quick Tests
```bash
# Test hybrid API
curl -X POST http://localhost:8001/api/autocomplete/hybrid \
  -H "Content-Type: application/json" \
  -d '{"prompt": "We are looking for"}'

# Test MLX model
curl -X POST http://localhost:8003/api/autocomplete/mlx \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Fun couple seeking"}'
```

### Load Testing
```bash
# Install hey
brew install hey

# Test hybrid endpoint
hey -n 100 -c 10 -m POST \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test prompt"}' \
  http://localhost:8001/api/autocomplete/hybrid
```

## 📈 Monitoring

### Logs
```bash
# API server logs
tail -f python/api_server.log

# MLX server logs  
tail -f python/mlx_server/mlx_server.log
```

### Metrics to Track
- Response times (target: <150ms hybrid, <100ms MLX)
- Cache hit rate (target: >80%)
- Error rate (target: <1%)
- Model load time

## 🔄 Deployment

### Production Checklist
1. Set production CORS origins
2. Enable HTTPS (use reverse proxy)
3. Add rate limiting
4. Set up logging aggregation
5. Configure health checks
6. Use process manager (PM2/systemd)

### Environment Variables
```bash
# Production .env
OLLAMA_PATH_API=http://ollama-server:11434/api
AUTOCOMPLETE_MODE=hybrid
LOG_LEVEL=info
WORKERS=4
```

Need help? Check the [Troubleshooting Guide](./TROUBLESHOOTING.md).