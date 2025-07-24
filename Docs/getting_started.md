# Getting Started - AI Bio Autocomplete

Welcome! This guide will help you set up the AI Bio Autocomplete app from scratch. No AI experience needed!

## 🎯 What is This App?

The AI Bio Autocomplete app helps you write personal bios by suggesting completions as you type. Think of it like Gmail's Smart Compose, but specifically trained for writing personal introductions. 

**Key features:**
- 🤖 **AI Suggestions** - Get intelligent text completions after typing 5+ words
- ✅ **Spell Check** - Built-in spell checker with click-to-fix
- 🚀 **Fast** - Suggestions appear in 100-150ms
- 🔒 **Privacy** - Everything runs locally on your computer

## 💡 How It Works (Simple Version)

1. You start typing your bio (e.g., "I am a fun loving person who...")
2. The app searches a database of similar bios
3. AI generates natural completions
4. Gray text appears showing suggestions
5. Press TAB to accept, or keep typing

## 📋 What You'll Need

### Software to Install
- **Node.js** v18+ - Runs the web interface ([Download](https://nodejs.org/))
- **Python** v3.9+ - Powers the AI backend ([Download](https://python.org/))
- **Git** - Downloads the code ([Download](https://git-scm.com/))
- **Ollama** - Runs AI models locally ([Download](https://ollama.ai/))

### Computer Requirements
- **Minimum**: 8GB RAM, 20GB free disk space
- **Recommended**: 16GB+ RAM, 50GB free disk space
- **For Training Custom Models**: Apple Silicon Mac (M1/M2/M3) with 16GB+ RAM

## 🚀 Step-by-Step Setup (15-20 minutes)

### Step 1: Download the Code
Open your terminal (Command Prompt on Windows) and run:
```bash
git clone <repository-url>
cd ai-train-llm
```
**What this does:** Downloads all the app files to your computer

### Step 2: Install the Web Interface
```bash
npm install
```
**What this does:** Installs all the components needed for the web interface (this may take 2-3 minutes)

### Step 3: Set Up the AI Backend
```bash
# Navigate to Python folder
cd python

# Create a virtual environment (keeps Python packages organized)
python3 -m venv venv

# Activate it (this changes your terminal prompt)
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows

# Install AI packages (this may take 3-5 minutes)
pip install -r requirements.txt

# Go back to main folder
cd ..
```
**What this does:** Sets up Python with all the AI libraries needed to run the backend

### Step 4: Install the AI Model
Open a **new terminal window** and run:
```bash
# Start Ollama (keep this running!)
ollama serve
```

In another terminal:
```bash
# Download the AI model (this is 7GB, may take 10-15 minutes)
ollama pull gemma3:12b

# Verify it downloaded
ollama list
```
**What this does:** Downloads the AI model that generates text suggestions. Think of it as the "brain" of the autocomplete.

### Step 5: Set Up the Bio Database
```bash
cd python
source venv/bin/activate  # Mac/Linux
# OR: venv\Scripts\activate  # Windows

python vector_db/setup_chromadb.py
cd ..
```
**What this does:** Creates a searchable database of 5,000+ example bios that helps generate relevant suggestions

### Step 6: Create Configuration File
```bash
echo "OLLAMA_PATH_API=http://127.0.0.1:11434/api" > .env.local
```
**What this does:** Tells the app where to find the AI model on your computer

## 🎮 Starting the App

You have two ways to run the app. Both are good - choose based on your needs:

### Option A: Hybrid Mode (Recommended for First Time)
This mode combines database search with AI generation for best results.

```bash
# Terminal 1: Start the backend
./start_hybrid.sh

# Terminal 2: Start the web interface
npm run dev
```

### Option B: Trained Mode (Faster, Requires Setup)
This mode uses a custom-trained AI model. Only works if you've trained a model (see Training Guide).

```bash
# Terminal 1: Start the AI model server
./start_trained.sh

# Terminal 2: Start the web interface
npm run dev
```

### Open the App
Once both commands are running, open your web browser and go to:
```
http://localhost:3000
```

You should see a simple form with a large text box. Start typing to see the magic! 🎉

## 🎮 How to Use the App

### Writing Your Bio
1. **Start Typing**: Click in the text box and start writing (e.g., "I am a creative person who...")
2. **Wait for Suggestions**: After about 5 words, you'll see gray text appear
3. **Accept or Ignore**:
   - Press `TAB` to accept the suggestion
   - Press `ESC` to dismiss it
   - Or just keep typing to ignore it

### Other Features
- **Spell Check**: Misspelled words appear with red underlines. Click them to see corrections.
- **Content Filter**: The app will warn you if it detects prohibited content (like external website references).
- **Clear Button**: Start over with a fresh bio anytime.

### Understanding the Modes
- **Hybrid Mode**: Like having a research assistant - it searches thousands of examples AND creates new suggestions
- **Trained Mode**: Like having a writing coach - it uses a custom AI model trained specifically on bio data

## ❓ Having Problems?

Don't worry! Check our comprehensive [Troubleshooting Guide](./TROUBLESHOOTING.md) for solutions to common issues like:
- "Ollama is not running"
- "Port already in use" 
- "Module not found" errors
- And many more...

## 📁 What's in This Project?

Here's a simple overview of the main folders:
```
ai-train-llm/
├── app/              # The web pages you see
├── components/       # UI elements (buttons, forms, etc.)
├── python/           # The AI backend that powers suggestions
├── data/             # Example bios used for training
├── models/           # Saved AI models (if you train your own)
└── Docs/             # All documentation (you are here!)
```

## 🎯 What's Next?

Now that you have the app running:

1. **Play with it!** Try writing different types of bios to see how the AI responds
2. **Learn more**: Read [How to Run the App](./RUNNING_THE_APP.md) for advanced options
3. **Train your own model**: Follow the [Training Guide](./TRAINING_GUIDE.md) (requires Mac with M1/M2/M3)
4. **Customize it**: Check the [Development Guide](./DEVELOPMENT_GUIDE.md) to modify the code

## 💡 Beginner Tips

- **Keep terminals open**: The app needs 2-3 terminal windows running
- **First suggestion is slow**: The AI needs to "warm up" (30-60 seconds for first suggestion)
- **Gray text not appearing?**: Make sure you typed at least 5 words and waited 2 seconds
- **Ollama issues?**: Make sure `ollama serve` is running in its own terminal

## 🎉 Congratulations!

You've successfully set up an AI-powered text completion system! This is the same technology behind tools like GitHub Copilot and ChatGPT, running entirely on your computer.

**Next recommended read**: [Running the App Guide](./RUNNING_THE_APP.md) to understand the different modes better.

---

*Remember: This documentation is for beginners. If something isn't clear, that's on us, not you! Feel free to open an issue for clarification.*