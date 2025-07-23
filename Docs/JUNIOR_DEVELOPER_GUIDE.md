# Complete Junior Developer Guide: AI Bio Autocomplete Project

This guide will walk you through setting up and running the AI Bio Autocomplete project from scratch. Even if you've never worked with Python or trained an LLM before, you'll be able to get this running!

## 🎯 What This Project Does

This is an AI-powered text autocomplete application that helps users write personal bios. It has two modes:
1. **Hybrid Mode**: Uses a vector database + Ollama AI (best quality, more variety)
2. **Trained Mode**: Uses a locally trained Llama model (fastest speed, 50-100ms)

## 📋 Prerequisites

### Required Software
1. **Node.js** (v18 or higher) - [Download](https://nodejs.org/)
2. **Python** (v3.9 or higher) - [Download](https://python.org/)
3. **Git** - [Download](https://git-scm.com/)
4. **Ollama** - [Download](https://ollama.ai/)

### Hardware Requirements
- **Minimum**: 8GB RAM, 10GB disk space
- **Recommended**: 16GB+ RAM (32GB for training)
- **For Training**: Apple Silicon Mac (M1/M2/M3) with 32GB+ RAM

## 🚀 Quick Start (Hybrid Mode)

This is the easiest way to get started. Follow these steps exactly:

### Step 1: Clone the Project
```bash
# Open Terminal (Mac) or Command Prompt (Windows)
git clone https://github.com/yourusername/ai-train-llm.git
cd ai-train-llm
```

### Step 2: Install Frontend Dependencies
```bash
# Install Node.js packages
npm install
```

### Step 3: Set Up Python Environment
```bash
# Navigate to Python directory
cd python

# Create virtual environment (isolates Python packages)
python3 -m venv venv

# Activate virtual environment
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt

# Go back to project root
cd ..
```

### Step 4: Set Up Ollama
```bash
# Install Ollama (if not already installed)
# On Mac: Download from https://ollama.ai/

# Start Ollama service
ollama serve

# In a new terminal, pull the AI model (this is 7GB, may take time)
ollama pull gemma3:12b

# Verify it's installed
ollama list
```

### Step 5: Set Up the Vector Database
```bash
# Make sure virtual environment is activated
cd python
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate     # Windows

# Run the setup script
python vector_db/setup_chromadb.py

# You should see: "ChromaDB setup complete! Indexed 5000 bios"
cd ..
```

### Step 6: Create Environment File
```bash
# Create .env.local file in project root
echo "OLLAMA_PATH_API=http://127.0.0.1:11434/api" > .env.local
```

### Step 7: Start Everything!
```bash
# Use the hybrid mode startup script
./start_hybrid.sh

# In a new terminal, start the frontend
npm run dev
```

### Step 8: Open the Application
Open your browser and go to: http://localhost:3000

🎉 **Congratulations!** The app is now running in hybrid mode!

## 📖 Understanding What's Running

When you start the application, here's what happens:

1. **Ollama Service** (Port 11434)
   - This is the AI brain that generates text
   - It runs the Gemma 3 12B model

2. **API Server** (Port 8001)
   - Connects the frontend to the AI services
   - Searches the vector database for similar bios
   - Combines database results with AI generation

3. **Next.js Frontend** (Port 3000)
   - The web interface you interact with
   - Handles spell checking, autocomplete, and more

## 🔧 How to Use the Application

1. **Start Typing**: Begin writing a bio in the text box
2. **Autocomplete**: After 5+ words, AI suggestions appear in gray
3. **Accept Suggestions**: Press TAB to accept
4. **Spell Check**: Misspelled words are underlined in red
5. **Click to Correct**: Click red words for suggestions

## 🛠️ Troubleshooting Common Issues

### "Ollama is not running"
```bash
# Start Ollama in a terminal
ollama serve

# Keep this terminal open!
```

### "API Server failed to start"
```bash
# Check if port 8001 is in use
lsof -i:8001

# Kill any process using it
kill -9 <PID>

# Try starting again
cd python && python api/api_server.py
```

### "Module not found" errors
```bash
# Make sure virtual environment is activated
cd python
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### Frontend won't start
```bash
# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
npm run dev
```

## 🏗️ Project Structure Explained

```
ai-train-llm/
├── app/                 # Next.js pages (frontend)
├── components/          # React components (UI pieces)
├── hooks/              # Custom React hooks (logic)
├── actions/            # Server-side API calls
├── python/             # All Python backend code
│   ├── api/           # API servers
│   ├── vector_db/     # Database code
│   └── mlx_training/  # Model training code
├── data/               # Training data
└── public/             # Static files
```

## 💡 Key Concepts for Beginners

### What's a Vector Database?
- Stores text as mathematical representations (vectors)
- Finds similar text super fast
- Like a smart search engine for bios

### What's Ollama?
- Runs AI models locally on your computer
- No internet needed after setup
- Generates human-like text

### What's an LLM?
- Large Language Model - an AI that understands and generates text
- Trained on lots of text to learn patterns
- Can complete sentences intelligently

### What's MLX?
- Apple's framework for running AI on Mac chips
- Super fast on M1/M2/M3 processors
- Used for training custom models

## 🎓 Next Steps

1. **Try Different Prompts**: Experiment with bio styles
2. **Adjust Settings**: Look in `components/Form.tsx` for customization
3. **Learn About Training**: See the [Local LLM Training Guide](./LOCAL_LLM_TRAINING_GUIDE.md)
4. **Explore the API**: Visit http://localhost:8001/docs

## 📞 Getting Help

- **Check Logs**: `tail -f python/api_server.log`
- **Status Check**: `python python/check_status.py`
- **Ask Questions**: Create an issue on GitHub

## 🎯 Daily Workflow

Every time you want to work on the project:

1. Start Ollama: `ollama serve`
2. Start backend: `./start_all_servers.sh`
3. Start frontend: `npm run dev`
4. Open browser: http://localhost:3000

Remember: Keep all terminal windows open while working!

---

You're now ready to use the AI Bio Autocomplete system! For advanced features like training your own model, check out the other guides in the Docs folder.