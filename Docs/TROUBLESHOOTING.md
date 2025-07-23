# Troubleshooting Guide

Quick solutions to common problems.

## 🚨 Setup Issues

### "Command not found" errors
```bash
# Make sure you're in the project directory
pwd  # Should show .../ai-train-llm

# For Python commands, activate virtual environment first
cd python && source venv/bin/activate
```

### "Module not found" errors

**Python:**
```bash
cd python
source venv/bin/activate
pip install -r requirements.txt
```

**Node.js:**
```bash
rm -rf node_modules package-lock.json
npm install
```

### "Port already in use"
```bash
# Find what's using the port
lsof -i :8001  # API server
lsof -i :8003  # MLX server
lsof -i :3000  # Frontend

# Kill the process
kill -9 <PID>

# Or kill by port
lsof -ti:8001 | xargs kill
```

## 🤖 Ollama Issues

### "Ollama is not running"
```bash
# Start Ollama (keep terminal open)
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

### "Model not found"
```bash
# Download the model (7GB)
ollama pull gemma3:12b

# List installed models
ollama list
```

### Ollama using too much memory
```bash
# Set memory limit
export OLLAMA_MAX_MEMORY=8GB
ollama serve
```

## 🔧 API Server Issues

### API server won't start
```bash
# Check logs
tail -50 python/api_server.log

# Common fixes:
cd python
source venv/bin/activate
pip install chromadb fastapi uvicorn

# Try manual start to see errors
python api/api_server.py
```

### "ChromaDB not found"
```bash
# Initialize the database
cd python
source venv/bin/activate
python vector_db/setup_chromadb.py
```

### Slow API responses
- First request is always slow (model loading)
- Check CPU usage: `top` or Activity Monitor
- Restart services if running for hours

## 🧠 MLX/Training Issues

### "MLX not found" (Mac only)
```bash
cd python
source venv/bin/activate
pip install mlx mlx-lm
```

### Training fails immediately
```bash
# Check you have training data
ls python/mlx_training/bio_mlx_improved/

# If missing, prepare it:
./prepare_training_data.sh
```

### "Out of memory" during training
```bash
# Edit start_training.sh
BATCH_SIZE=1  # Reduce from 4
LORA_RANK=8   # Reduce from 16
```

### Model not loading
```bash
# Check model exists
ls models/bio-sentence-llama3-lora/

# If missing, use fallback or train new model
./start_training.sh
```

## 💻 Frontend Issues

### "Cannot GET /"
```bash
# Wrong terminal - make sure you're in project root
cd /path/to/ai-train-llm
npm run dev
```

### Autocomplete not working
1. Check browser console (F12)
2. Verify backend is running:
   ```bash
   curl http://localhost:8001/  # Hybrid
   curl http://localhost:8003/  # Trained
   ```
3. Check mode in `.env.local`:
   ```bash
   cat .env.local  # Should show AUTOCOMPLETE_MODE
   ```

### Spell check not working
- Clear browser cache and reload
- Check dictionary files exist:
  ```bash
  ls public/dictionaries/en_US/
  ```

## 🔄 Mode Switching Issues

### App stuck in wrong mode
```bash
# Check current mode
cat .env.local

# Force mode change
echo "AUTOCOMPLETE_MODE=hybrid" > .env.local  # or 'trained'

# Restart frontend
# Ctrl+C in npm terminal, then:
npm run dev
```

### Both servers running
```bash
# This is OK but wastes resources
# Stop the one you don't need:
lsof -ti:8001 | xargs kill  # Stop hybrid
lsof -ti:8003 | xargs kill  # Stop trained
```

## 🐛 General Debugging

### Check all services
```bash
# Quick health check
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:8001/           # API server
curl http://localhost:8003/           # MLX server
curl http://localhost:3000/           # Frontend
```

### View all logs
```bash
# In separate terminals:
tail -f python/api_server.log
tail -f python/mlx_server/mlx_server.log
# Frontend logs show in browser console
```

### Reset everything
```bash
# Stop all services (Ctrl+C in all terminals)

# Clear Python cache
find . -type d -name __pycache__ -exec rm -r {} +

# Clear Node cache  
rm -rf .next node_modules

# Reinstall
npm install
cd python && pip install -r requirements.txt

# Start fresh
./start_hybrid.sh
npm run dev
```

## 📞 Getting Help

If these solutions don't work:

1. **Check logs** for specific error messages
2. **Search issues** on GitHub
3. **Open new issue** with:
   - Your OS and hardware
   - Exact error message
   - What you tried
   - Relevant log snippets

## 💡 Pro Tips

- Always check logs first
- Keep terminals open while running
- Restart services if acting weird
- First response is slow (normal)
- Use Activity Monitor (Mac) or Task Manager (Windows) to check resources

Remember: Most issues are from:
1. Services not running
2. Wrong directory
3. Virtual environment not activated
4. Ports already in use

Start there and you'll solve 90% of problems!