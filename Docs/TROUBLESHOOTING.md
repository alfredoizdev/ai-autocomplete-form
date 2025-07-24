# Troubleshooting Guide

Don't panic! Most issues have simple fixes. This guide is organized by when problems occur.

## 🏁 During Initial Setup

### "Command not found" errors

**What it looks like:**
```
bash: npm: command not found
```

**Why this happens:** The required software isn't installed yet.

**Fix:**
1. Make sure you've installed all requirements:
   - Node.js from https://nodejs.org
   - Python from https://python.org
   - Git from https://git-scm.com

2. Restart your terminal after installing

3. Verify installations:
   ```bash
   node --version    # Should show v18.x.x or higher
   python3 --version # Should show 3.9.x or higher
   git --version     # Should show version info
   ```

### "Module not found" or "Cannot find package" errors

**What it looks like:**
```
Error: Cannot find module 'react'
ModuleNotFoundError: No module named 'fastapi'
```

**Why this happens:** Dependencies aren't installed yet.

**Fix for Node.js errors:**
```bash
# Make sure you're in the main project folder
cd /path/to/ai-train-llm

# Clean install
rm -rf node_modules package-lock.json
npm install
```

**Fix for Python errors:**
```bash
# Navigate to Python folder
cd python

# Activate virtual environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### "Port already in use" error

**What it looks like:**
```
Error: listen EADDRINUSE: address already in use :::3000
```

**Why this happens:** Another program is using that port, or the app didn't shut down properly.

**Quick fix:**
```bash
# See what's using the port
lsof -i :3000  # For frontend
lsof -i :8001  # For API server
lsof -i :8003  # For MLX server

# Stop it (replace PID with the number from above)
kill -9 PID
```

**Nuclear option (stops all):**
```bash
pkill -f node
pkill -f python
```

## 🚀 When Starting the App

### "Ollama is not running"

**What it looks like:**
```
Error: Ollama service not available at http://127.0.0.1:11434
```

**Why this happens:** Ollama needs to be running in the background.

**Fix:**
1. Open a new terminal window
2. Run: `ollama serve`
3. Keep this terminal open!
4. You should see: "Ollama is running on http://127.0.0.1:11434"

### "Model not found: gemma3:12b"

**What it looks like:**
```
Error: model 'gemma3:12b' not found
```

**Why this happens:** The AI model hasn't been downloaded yet.

**Fix:**
```bash
# Download the model (this is 7GB, takes 10-15 minutes)
ollama pull gemma3:12b

# Verify it's installed
ollama list
# Should show: gemma3:12b
```

### "Cannot connect to backend" or "API server not responding"

**What it looks like:**
- Autocomplete doesn't work
- Gray suggestions never appear
- Browser console shows connection errors

**Why this happens:** The backend server isn't running.

**Fix:**
1. Make sure you ran the startup script:
   ```bash
   ./start_hybrid.sh   # or ./start_trained.sh
   ```

2. Check if it's actually running:
   ```bash
   curl http://localhost:8001/
   # Should return: {"status":"ok"}
   ```

3. If not, check the logs:
   ```bash
   tail -f python/api_server.log
   ```

## 🖥️ While Using the App

### No gray suggestions appearing

**Common causes and fixes:**

1. **Haven't typed enough words**
   - Need at least 5 words
   - Try: "I am a fun loving person who"

2. **Typing too fast**
   - App waits 1.5 seconds after you stop
   - Pause after typing

3. **Backend not running**
   - Check both terminals are still running
   - Look for error messages

4. **Wrong mode set**
   ```bash
   # Check current mode
   cat .env.local
   # Should show AUTOCOMPLETE_MODE=hybrid or trained
   ```

### Suggestions are very slow (30+ seconds)

**Why this happens:** First suggestion loads the AI model.

**Fix:**
1. Wait for the first one - it gets faster!
2. Check Activity Monitor (Mac) for high CPU usage
3. Close other heavy applications
4. Restart the backend services

### "ChromaDB not initialized" error

**What it looks like:**
```
Error: Collection 'bio_embeddings' not found
```

**Why this happens:** The bio database wasn't set up.

**Fix:**
```bash
cd python
source venv/bin/activate
python vector_db/setup_chromadb.py
# Should see: "ChromaDB setup completed successfully"
```

## 🧠 Training Issues (Mac Only)

### "MLX not found"

**What it looks like:**
```
ModuleNotFoundError: No module named 'mlx'
```

**Why this happens:** MLX (Apple's ML framework) isn't installed.

**Fix:**
```bash
cd python
source venv/bin/activate
pip install mlx mlx-lm
```

### Training fails immediately

**What it looks like:**
```
Error: No training data found in lookingfor_hq/
```

**Why this happens:** Training data hasn't been prepared.

**Fix:**
```bash
# Prepare the data first
./prepare_hq_data_fast.sh

# Should see: "Training data ready!"
# Then try training again
./start_training_mlx_community.sh
```

### "Out of memory" during training

**What it looks like:**
```
RuntimeError: MPS backend out of memory
```

**Why this happens:** Training uses lots of memory.

**Fix:**
1. Close all other apps
2. Edit `start_training_mlx_community.sh`:
   ```bash
   BATCH_SIZE=2    # Reduce from 4
   NUM_LAYERS=16   # Reduce from 24
   ```
3. Restart your Mac and try again

### Trained mode says "Model not found"

**Why this happens:** No trained model exists yet.

**Check for models:**
```bash
ls models/
# Should show folders like: lookingfor-llama3-3b-hq-lora/
```

**If empty, train a model:**
```bash
./prepare_hq_data_fast.sh
./start_training_mlx_community.sh
```

## 🌐 Browser Issues

### Page shows "Cannot GET /"

**Why this happens:** You're in the wrong folder or frontend isn't running.

**Fix:**
```bash
# Make sure you're in the project root
pwd
# Should end with: /ai-train-llm

# Start the frontend
npm run dev
# Should see: Ready - started server on http://localhost:3000
```

### Spell check not working

**What it looks like:**
- No red underlines on misspelled words
- Clicking misspelled words does nothing

**Fix:**
1. Hard refresh the page: `Cmd+Shift+R` (Mac) or `Ctrl+Shift+R` (Windows)
2. Clear browser cache
3. Check console for errors (F12)

## 🔄 Switching Modes

### App using wrong mode after switching

**Why this happens:** Browser cached the old mode.

**Fix:**
1. Stop all services (Ctrl+C in terminals)
2. Start the mode you want:
   ```bash
   ./start_hybrid.sh   # or ./start_trained.sh
   ```
3. Hard refresh browser: `Cmd+Shift+R`

## 🆘 Emergency Fixes

### "Nothing is working!"

**The Nuclear Reset:**
```bash
# 1. Stop everything
pkill -f node
pkill -f python
pkill -f ollama

# 2. Clear all caches
rm -rf .next node_modules
rm -rf python/__pycache__

# 3. Reinstall everything
npm install
cd python && pip install -r requirements.txt && cd ..

# 4. Start fresh
ollama serve  # In terminal 1
./start_hybrid.sh  # In terminal 2
npm run dev  # In terminal 3
```

### Check if anything is actually running

**Quick health check script:**
```bash
echo "Checking services..."
echo -n "Ollama: "
curl -s http://localhost:11434 && echo "✓ Running" || echo "✗ Not running"
echo -n "API Server: "
curl -s http://localhost:8001 && echo "✓ Running" || echo "✗ Not running"
echo -n "MLX Server: "
curl -s http://localhost:8003 && echo "✓ Running" || echo "✗ Not running"
echo -n "Frontend: "
curl -s http://localhost:3000 && echo "✓ Running" || echo "✗ Not running"
```

## 💡 Golden Rules

1. **Read the error message** - It usually tells you what's wrong
2. **Check the logs** - They have more details
3. **One terminal per service** - Don't close them!
4. **Patience with first run** - Things need to warm up
5. **When in doubt, restart** - Solves many mysterious issues

## 📚 Still Stuck?

If nothing here helps:

1. Take a screenshot of the error
2. Copy the error message
3. Note what you were doing when it happened
4. Check the project's GitHub issues
5. Open a new issue with all this info

---

*Remember: Every developer faces these issues. You're not alone, and it's not your fault! The fact that you're troubleshooting means you're learning.*