# Running the App - Operating Guide

Already set up? Great! This guide explains how to run the app day-to-day and understand the different modes.

## 🎯 Understanding the Two Modes

Think of the app like a restaurant with two chefs:

### 🔄 Hybrid Mode (The Research Chef)
This mode is like a chef who:
- Looks through a cookbook of 5,000+ recipes (vector search)
- Then creates something new inspired by what they found (AI generation)
- Takes a bit longer but creates more varied dishes

**Perfect for**: General use, trying different styles, exploring possibilities

### 🚀 Trained Mode (The Specialist Chef)  
This mode is like a chef who:
- Has practiced making one type of cuisine perfectly (fine-tuned model)
- Doesn't need to look at recipes - it's all in their head
- Faster and more consistent, but less variety

**Perfect for**: Consistent style, faster responses, production use

## 🚀 Quick Start Commands

### Starting Hybrid Mode (Recommended First)
Open two terminal windows:

**Terminal 1 - Backend:**
```bash
./start_hybrid.sh
```
You'll see: `INFO: Application startup complete` when ready

**Terminal 2 - Frontend:**
```bash
npm run dev
```
You'll see: `Ready in X seconds - http://localhost:3000`

### Starting Trained Mode
First, make sure you have a trained model (see [Training Guide](./TRAINING_GUIDE.md)).

**Terminal 1 - AI Model:**
```bash
./start_trained.sh
```
You'll see: `Model loaded successfully` when ready

**Terminal 2 - Frontend:**
```bash
npm run dev
```

## 🔍 What's Actually Happening?

### When You Start Hybrid Mode:
1. **Checks Ollama** - Makes sure your AI model (Gemma) is available
2. **Starts API Server** - Launches the Python backend on port 8001
3. **Loads Database** - Connects to the bio examples database
4. **Sets Mode** - Tells the frontend to use hybrid autocomplete

### When You Start Trained Mode:
1. **Loads Custom Model** - Loads your fine-tuned Llama model into memory
2. **Starts MLX Server** - Launches the model server on port 8003
3. **Optimizes for Speed** - Prepares model for fast inference
4. **Sets Mode** - Tells the frontend to use the trained model

## 📊 Checking If Everything's Working

### Quick Health Check
Visit these URLs in your browser:
- **Frontend**: http://localhost:3000 (should show the app)
- **Hybrid API**: http://localhost:8001/docs (should show API documentation)
- **Ollama**: http://localhost:11434 (should show "Ollama is running")

### Watching the Logs (See What's Happening)
Keep these running in separate terminals to see real-time activity:

**Watch API requests (Hybrid mode):**
```bash
tail -f python/api_server.log
```
You'll see each autocomplete request as you type!

**Watch model server (Trained mode):**
```bash
tail -f python/mlx_server/mlx_server.log
```

### Is Something Not Working?
Run this diagnostic command:
```bash
# Shows all running services
ps aux | grep -E "python|node|ollama" | grep -v grep
```

## 🔄 Switching Between Modes

Want to try the other mode? It's easy:

1. **Stop current services**: Press `Ctrl+C` in both terminal windows
2. **Start the other mode**: Run the other startup script
3. **Refresh your browser**: The app will automatically use the new mode

The startup scripts handle all the configuration for you!

## ⚙️ Adjusting Settings (Optional)

### Making Suggestions Appear Faster/Slower
Edit `.env.local` and add:
```bash
# Faster suggestions (may be less accurate)
DEBOUNCE_DELAY=500

# Slower suggestions (more time to think)
DEBOUNCE_DELAY=3000
```

### Want More/Fewer Suggestions?
The app is configured for optimal performance, but advanced users can modify settings in:
- `python/api/api_server.py` - Backend settings
- `hooks/useFormAutocomplete.tsx` - Frontend behavior

For detailed configuration options, see the [Configuration Guide](./CONFIGURATION.md).

## 🚦 Running in Production

### For Personal Use (Always On)
Use PM2 to keep the app running even after reboot:
```bash
# Install PM2
npm install -g pm2

# Start and save
pm2 start ./start_hybrid.sh --name "ai-bio-backend"
pm2 start npm --name "ai-bio-frontend" -- start
pm2 save
pm2 startup
```

### For Team/Public Use
See the [Development Guide](./DEVELOPMENT_GUIDE.md) for Docker deployment options.

## 📈 Mode Comparison Chart

| Aspect | Hybrid Mode | Trained Mode |
|--------|-------------|--------------|
| **Setup Difficulty** | ⭐ Easy | ⭐⭐⭐ Requires training |
| **Response Quality** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Very Good |
| **Speed** | ⭐⭐⭐⭐ Fast (150ms) | ⭐⭐⭐⭐⭐ Very Fast (100ms) |
| **Variety** | ⭐⭐⭐⭐⭐ High | ⭐⭐⭐ Moderate |
| **Best For** | Exploring, variety | Consistency, speed |

## 🆘 Quick Fixes

**App not responding?**
1. Check both terminals are still running
2. Refresh your browser
3. Check [Troubleshooting Guide](./TROUBLESHOOTING.md)

**Want to stop everything?**
Press `Ctrl+C` in all terminal windows

**Want it to start automatically?**
Use the PM2 commands above

## 📚 Next Steps

- **New to AI?** Learn how it works in our [Architecture Guide](./ARCHITECTURE.md)
- **Want to train a model?** Follow the [Training Guide](./TRAINING_GUIDE.md)
- **Having issues?** Check the [Troubleshooting Guide](./TROUBLESHOOTING.md)

---

*Pro tip: Keep the log terminal open while using the app - it's fascinating to watch the AI think!*