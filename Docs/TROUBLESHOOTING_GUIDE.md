# Comprehensive Troubleshooting Guide

This guide covers common issues and their solutions. Each problem includes symptoms, causes, and step-by-step fixes.

## 🔍 Quick Diagnostic

Run this first to identify issues:
```bash
python python/check_status.py
```

## 📋 Table of Contents

1. [Setup Issues](#setup-issues)
2. [Ollama Problems](#ollama-problems)
3. [API Server Errors](#api-server-errors)
4. [Frontend Issues](#frontend-issues)
5. [Training Problems](#training-problems)
6. [Performance Issues](#performance-issues)
7. [Integration Errors](#integration-errors)

---

## Setup Issues

### Problem: "Module not found" Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'chromadb'
ModuleNotFoundError: No module named 'fastapi'
```

**Solution:**
```bash
# Make sure you're in the python directory
cd python

# Activate virtual environment
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt

# Verify installation
pip list | grep chromadb
```

### Problem: Virtual Environment Not Working

**Symptoms:**
- `venv/bin/activate: No such file or directory`
- Python packages installing globally

**Solution:**
```bash
# Remove old venv
rm -rf python/venv

# Create new venv
cd python
python3 -m venv venv

# Activate it
source venv/bin/activate

# Verify it's active (should show venv path)
which python
```

### Problem: Port Already in Use

**Symptoms:**
```
[Errno 48] Address already in use
OSError: [Errno 98] Address already in use
```

**Solution:**
```bash
# Find what's using the port
lsof -i:8001  # For API server
lsof -i:8003  # For MLX server
lsof -i:3000  # For Next.js

# Kill the process
kill -9 <PID>

# Or kill all Python processes (careful!)
pkill -f python
```

---

## Ollama Problems

### Problem: Ollama Not Starting

**Symptoms:**
- `Failed to connect to Ollama`
- `Connection refused on port 11434`

**Solution:**
```bash
# Check if Ollama is installed
which ollama

# If not installed, download from https://ollama.ai/

# Start Ollama service
ollama serve

# In another terminal, verify it's running
curl http://localhost:11434/api/tags
```

### Problem: Model Not Found

**Symptoms:**
- `model 'gemma3:12b' not found`
- `pull model gemma3:12b first`

**Solution:**
```bash
# Pull the model (7GB download)
ollama pull gemma3:12b

# Verify it's downloaded
ollama list

# Test the model
ollama run gemma3:12b "Hello"
```

### Problem: Ollama Running Out of Memory

**Symptoms:**
- Slow responses
- System freezing
- `out of memory` errors

**Solution:**
```bash
# Check memory usage
ollama ps

# Stop all models
ollama stop gemma3:12b

# Restart with memory limit
OLLAMA_MAX_LOADED_MODELS=1 ollama serve
```

---

## API Server Errors

### Problem: ChromaDB Not Initialized

**Symptoms:**
- `Collection 'bio_collection' not found`
- Empty autocomplete results

**Solution:**
```bash
cd python
source venv/bin/activate

# Run setup script
python vector_db/setup_chromadb.py

# Verify database exists
ls -la chroma_db/
```

### Problem: API Server Crashing

**Symptoms:**
- Server stops randomly
- `Segmentation fault`

**Solution:**
```bash
# Check Python version (needs 3.9+)
python --version

# Run with more logging
cd python
python api/api_server.py --log-level debug

# Check for corrupted database
rm -rf chroma_db
python vector_db/setup_chromadb.py
```

### Problem: Slow API Responses

**Symptoms:**
- Requests taking >1 second
- Timeout errors

**Solution:**
1. **Check Ollama connection:**
   ```bash
   curl -X POST http://localhost:11434/api/generate \
     -d '{"model": "gemma3:12b", "prompt": "test"}'
   ```

2. **Reduce vector search results:**
   ```python
   # In api_server.py
   NUM_SIMILAR_BIOS = 5  # Instead of 10
   ```

3. **Pre-load the model:**
   ```bash
   # Keep model in memory
   ollama run gemma3:12b "warm up"
   ```

---

## Frontend Issues

### Problem: Autocomplete Not Triggering

**Symptoms:**
- No gray suggestions appearing
- Tab key not working

**Solution:**
1. **Check browser console** (F12):
   - Look for red errors
   - Check network tab for API calls

2. **Verify word count:**
   ```javascript
   // Need 5+ words in current sentence
   // Check hooks/useFormAutocomplete.tsx
   const MIN_WORDS = 5;
   ```

3. **Clear browser cache:**
   - Hard refresh: Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows)

### Problem: Spell Check Not Working

**Symptoms:**
- No red underlines
- Dictionary not loading

**Solution:**
```bash
# Check dictionary files exist
ls public/dictionaries/

# If missing, download them:
cd public/dictionaries
wget https://raw.githubusercontent.com/wooorm/dictionaries/main/dictionaries/en/index.aff
wget https://raw.githubusercontent.com/wooorm/dictionaries/main/dictionaries/en/index.dic
```

### Problem: Build Errors

**Symptoms:**
- `npm run build` fails
- TypeScript errors

**Solution:**
```bash
# Clean install
rm -rf node_modules package-lock.json
npm install

# Check for TypeScript errors
npm run build

# If persist, check Node version (need 18+)
node --version
```

---

## Training Problems

### Problem: MLX Not Installing

**Symptoms:**
- `No module named 'mlx'`
- Installation fails on non-Apple Silicon

**Solution:**
```bash
# Verify you have Apple Silicon
python -c "import platform; print(platform.processor())"
# Should show 'arm' or 'arm64'

# Install MLX
pip install --upgrade pip
pip install mlx mlx-lm
```

### Problem: Training Out of Memory

**Symptoms:**
- `Metal out of memory`
- Training crashes

**Solution:**
1. **Reduce batch size:**
   ```yaml
   # In config.yaml
   training:
     batch_size: 1
     gradient_accumulation: 8
   ```

2. **Reduce sequence length:**
   ```yaml
   data:
     max_seq_length: 128  # Instead of 256
   ```

3. **Use smaller model:**
   ```yaml
   model: "mlx-community/Qwen2.5-3B-Instruct-4bit"
   ```

### Problem: Poor Model Quality

**Symptoms:**
- Nonsensical completions
- Repetitive output
- Grammar errors

**Solution:**
1. **Check training data:**
   ```bash
   # Inspect prepared data
   head -50 python/mlx_server/data/train.jsonl
   ```

2. **Train longer:**
   ```yaml
   training:
     num_epochs: 5  # Instead of 3
     max_steps: 2000  # Instead of 1000
   ```

3. **Adjust temperature during inference:**
   ```python
   # In mlx_model_server.py
   temperature = 0.7  # Lower for more focused output
   ```

---

## Performance Issues

### Problem: High Memory Usage

**Symptoms:**
- System slowing down
- Beach ball of death (Mac)

**Solution:**
```bash
# Check memory usage
top -o mem

# Restart services
./stop_all_servers.sh  # Create this script
./start_all_servers.sh

# Limit Ollama models
export OLLAMA_MAX_LOADED_MODELS=1
```

### Problem: Slow Autocomplete

**Symptoms:**
- 1+ second delays
- UI freezing

**Solution:**
1. **Enable caching:**
   ```javascript
   // Already enabled in ai-text-streaming.ts
   // Check cache is working
   ```

2. **Use MLX model:**
   ```bash
   # Faster than Ollama
   echo "NEXT_PUBLIC_USE_FINETUNED_MODEL=true" >> .env.local
   ```

3. **Reduce debounce time:**
   ```javascript
   // In useFormAutocomplete.tsx
   const DEBOUNCE_DELAY = 800;  // Instead of 1500
   ```

---

## Integration Errors

### Problem: Services Can't Connect

**Symptoms:**
- `ECONNREFUSED`
- `Failed to fetch`

**Solution:**
1. **Check all services running:**
   ```bash
   curl http://localhost:11434/api/tags  # Ollama
   curl http://localhost:8001/api/stats  # API Server
   curl http://localhost:8003/health     # MLX Server
   ```

2. **Check firewall:**
   ```bash
   # Mac: Check Security & Privacy settings
   # May need to allow connections
   ```

3. **Use 127.0.0.1 instead of localhost:**
   ```bash
   # In .env.local
   OLLAMA_PATH_API=http://127.0.0.1:11434/api
   ```

### Problem: CORS Errors

**Symptoms:**
- `Access-Control-Allow-Origin` errors
- Blocked by CORS policy

**Solution:**
```python
# In api_server.py, CORS is already configured
# If issues persist, check:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🚨 Emergency Recovery

If everything is broken:

```bash
# 1. Stop everything
pkill -f python
pkill -f node
pkill -f ollama

# 2. Clean up
cd ai-train-llm
rm -rf python/venv
rm -rf node_modules
rm -rf python/chroma_db
rm -rf python/mlx_server/models/bio-phi3-lora

# 3. Fresh install
npm install
cd python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Reinitialize
python vector_db/setup_chromadb.py

# 5. Start fresh
cd ..
./start_all_servers.sh
npm run dev
```

---

## 📞 Getting More Help

1. **Check Logs:**
   ```bash
   tail -f python/api_server.log
   tail -f python/mlx_server.log
   ```

2. **Enable Debug Mode:**
   ```bash
   export DEBUG=true
   ```

3. **System Information:**
   ```bash
   # Helpful for bug reports
   system_profiler SPSoftwareDataType
   python --version
   node --version
   npm --version
   ```

4. **Community Help:**
   - Create detailed GitHub issue
   - Include error messages
   - Share system info
   - Describe steps to reproduce

Remember: Most issues are configuration-related. Double-check your setup before assuming there's a bug!