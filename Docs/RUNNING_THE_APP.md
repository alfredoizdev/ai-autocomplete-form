# Running the App

This guide explains how to run the AI Bio Autocomplete app in different modes.

## 🎯 Available Modes

### 1. Hybrid Mode (Recommended)
- **Best for**: High-quality suggestions with variety
- **How it works**: Combines vector search with AI generation
- **Response time**: 100-150ms
- **Requirements**: Ollama running

### 2. Trained Mode  
- **Best for**: Highest quality completions with fast response times
- **How it works**: Uses HIGH-QUALITY fine-tuned Llama-3.2-3B model
- **Response time**: 100-150ms
- **Requirements**: Trained MLX model (4.5k+ grammar-filtered examples)

## 🚀 Starting the App

### Hybrid Mode
```bash
# Terminal 1: Start backend services
./start_hybrid.sh

# Terminal 2: Start frontend
npm run dev
```

What happens:
1. Sets `AUTOCOMPLETE_MODE=hybrid` automatically
2. Starts API server on port 8001
3. Checks that Ollama is running
4. Shows real-time logs

### Trained Mode
```bash
# Terminal 1: Start MLX server  
./start_trained.sh

# Terminal 2: Start frontend
npm run dev
```

What happens:
1. Sets `AUTOCOMPLETE_MODE=trained` automatically
2. Starts MLX server on port 8003
3. Loads HIGH-QUALITY Llama-3.2-3B model (grammar-filtered, 2000 iterations)
4. Shows real-time logs

## 📊 Monitoring Services

### Check What's Running
```bash
# Check all services
lsof -i :8001  # API Server (hybrid)
lsof -i :8003  # MLX Server (trained)
lsof -i :11434 # Ollama
lsof -i :3000  # Frontend
```

### View Logs
```bash
# API server logs (hybrid mode)
tail -f python/api_server.log

# MLX server logs (trained mode)
tail -f python/mlx_server/mlx_server.log
```

### API Documentation
- Hybrid Mode: http://localhost:8001/docs
- Frontend: http://localhost:3000

## 🛠️ Manual Mode Control

If you prefer manual control over automatic scripts:

### Manual Hybrid Mode
```bash
# Set mode
echo "AUTOCOMPLETE_MODE=hybrid" >> .env.local

# Start Ollama
ollama serve

# Start API server
cd python
source venv/bin/activate  
python api/api_server.py

# Start frontend
npm run dev
```

### Manual Trained Mode
```bash
# Set mode
echo "AUTOCOMPLETE_MODE=trained" >> .env.local

# Start MLX server
cd python/mlx_server
source ../venv/bin/activate
python mlx_model_server.py

# Start frontend  
npm run dev
```

## ⚙️ Configuration

### Environment Variables
```bash
# Required
OLLAMA_PATH_API=http://127.0.0.1:11434/api

# Mode selection (set by scripts)
AUTOCOMPLETE_MODE=hybrid    # or 'trained'
```

### Performance Tuning

**Hybrid Mode** (`python/api/api_server.py`):
```python
NUM_SIMILAR_BIOS = 10        # Vector search results
MIN_COMPLETION_LENGTH = 8    # Min words in suggestion
TEMPERATURES = [0.7, 0.9]    # AI creativity levels
```

**Frontend** (`hooks/useFormAutocomplete.tsx`):
```javascript
DEBOUNCE_DELAY = 1500       # Typing delay (ms)
MIN_WORDS = 5               # Words before suggestions
```

## 🔄 Switching Modes

To switch between modes:

1. **Stop current services** (Ctrl+C in terminals)
2. **Run the other startup script**
3. **Restart frontend** if needed

The scripts automatically update your `.env.local` file.

## 🚦 Production Deployment

### Using PM2
```bash
# Install PM2
npm install -g pm2

# Start services
pm2 start ./start_hybrid.sh --name "bio-backend"
pm2 start npm --name "bio-frontend" -- start

# Monitor
pm2 monit
```

### Using Docker
```bash
# Build and run
docker-compose up -d

# Check status
docker-compose ps
```

## 📈 Performance Comparison

| Feature | Hybrid Mode | Trained Mode |
|---------|------------|--------------|
| Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Speed | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Variety | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Setup | Easy | Requires training |

## 🆘 Troubleshooting

### Services Won't Start
```bash
# Kill any stuck processes
pkill -f "python.*api_server"
pkill -f "python.*mlx_model_server"

# Clear ports
lsof -ti:8001 | xargs kill
lsof -ti:8003 | xargs kill
```

### Slow Responses
- First response is always slower (model loading)
- Check if other apps are using CPU/memory
- Restart services if they've been running long

### Mode Not Switching
```bash
# Manually check/set mode
cat .env.local
echo "AUTOCOMPLETE_MODE=hybrid" > .env.local
```

Need more help? See the [Troubleshooting Guide](./TROUBLESHOOTING.md).