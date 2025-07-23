# Getting Started - AI Bio Autocomplete

This guide will help you set up and run the AI Bio Autocomplete app. Even if you're new to AI or web development, you'll be able to get this running!

## 📋 What You Need

### Software Requirements
- **Node.js** v18+ ([Download](https://nodejs.org/))
- **Python** v3.9+ ([Download](https://python.org/))
- **Git** ([Download](https://git-scm.com/))
- **Ollama** ([Download](https://ollama.ai/))

### Hardware Requirements
- **Minimum**: 8GB RAM, 20GB disk space
- **Recommended**: 16GB+ RAM, 50GB disk space
- **For Training**: Apple Silicon Mac with 16GB+ RAM (32GB+ recommended)

## 🚀 Quick Setup (15 minutes)

### Step 1: Clone the Project
```bash
git clone <repository-url>
cd ai-train-llm
```

### Step 2: Install Dependencies
```bash
# Install frontend packages
npm install

# Set up Python environment
cd python
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

### Step 3: Set Up Ollama
```bash
# Start Ollama service (keep this running)
ollama serve

# In a new terminal, download the AI model (7GB)
ollama pull gemma3:12b
```

### Step 4: Initialize Database
```bash
cd python
source venv/bin/activate
python vector_db/setup_chromadb.py
cd ..
```

### Step 5: Configure Environment
```bash
# Create environment file
echo "OLLAMA_PATH_API=http://127.0.0.1:11434/api" > .env.local
```

### Step 6: Choose Your Mode & Start

#### Option A: Hybrid Mode (Recommended)
Best quality with vector search + AI generation
```bash
./start_hybrid.sh      # Starts backend services
# In new terminal:
npm run dev           # Starts frontend
```

#### Option B: Trained Mode (Best Quality)
Uses fine-tuned Llama model for high-quality completions
```bash
./start_trained.sh    # Starts MLX server with HIGH-QUALITY model
# In new terminal:
npm run dev          # Starts frontend
```

Note: The trained mode uses the HIGH-QUALITY Llama-3.2-3B model fine-tuned on grammar-filtered LookingFor dataset (4.5k+ examples, 2000 iterations, learning rate 1e-5) for superior quality. Falls back to standard models if not available.

### Step 7: Open the App
Navigate to http://localhost:3000 in your browser

## 🎮 Using the App

### Basic Features
1. **Type Your Bio**: Start typing in the text area
2. **AI Suggestions**: After ~5 words, gray text appears with suggestions
3. **Accept Suggestions**: Press TAB to accept
4. **Spell Check**: Click red underlined words for corrections
5. **Safety Filter**: Automatically detects prohibited content

### Modes Explained
- **Hybrid Mode**: Searches similar bios + generates new content (balanced approach)
- **Trained Mode**: Uses HIGH-QUALITY fine-tuned 3B model (best quality, 100-150ms, grammar-filtered dataset)

### Keyboard Shortcuts
- `TAB` - Accept AI suggestion
- `ESC` - Dismiss suggestion
- Click misspelled words - See corrections

## 🔧 Common Setup Issues

### "Ollama is not running"
```bash
# Make sure Ollama is running in a terminal:
ollama serve
```

### "Port already in use"
```bash
# Find and kill the process:
lsof -ti:8001 | xargs kill  # API server
lsof -ti:8003 | xargs kill  # MLX server
```

### "Module not found" Python errors
```bash
# Activate virtual environment first:
cd python && source venv/bin/activate
pip install -r requirements.txt
```

### "Cannot find module" Node errors
```bash
rm -rf node_modules package-lock.json
npm install
```

## 📁 Project Structure

```
ai-train-llm/
├── app/              # Frontend pages
├── components/       # React components  
├── hooks/           # Custom React hooks
├── python/          # Backend services
│   ├── api/         # API server
│   ├── mlx_server/  # ML model server
│   └── vector_db/   # Database code
├── data/            # Training data
└── models/          # Trained models
```

## 🎯 Next Steps

1. **Run the app**: Follow the setup above
2. **Customize**: Edit prompts in `actions/ai-text.ts`
3. **Train a model**: See [Training Guide](./TRAINING_GUIDE.md)
4. **Deploy**: See production deployment in [Running the App](./RUNNING_THE_APP.md)

## 💡 Tips for Success

- Keep terminal windows open while running
- Check logs if something fails: `tail -f python/api_server.log`
- The first AI response may be slow as models load
- Hybrid mode gives best results, trained mode is fastest

Need help? Check the [Troubleshooting Guide](./TROUBLESHOOTING.md) or open an issue!