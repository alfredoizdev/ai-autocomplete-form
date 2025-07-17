# Getting Started with AI Bio Generator

Welcome! This guide will help you understand and use the AI Bio Generator app, whether you're completely new or looking to customize it for your needs.

## Table of Contents
- [What is this app?](#what-is-this-app)
- [Quick Start (5 minutes)](#quick-start-5-minutes)
- [How Does It Work?](#how-does-it-work)
- [Using the App](#using-the-app)
- [Customizing Your Experience](#customizing-your-experience)
- [Understanding the Features](#understanding-the-features)
- [Privacy & Safety](#privacy--safety)
- [Common Questions](#common-questions)
- [Troubleshooting](#troubleshooting)

## What is this app?

The AI Bio Autocomplete is a sophisticated text completion system designed for the swinger community. It helps you write compelling personal bios by:

- 🤖 **AI-powered suggestions** using hybrid vector search + LLM generation
- ⚡ **Lightning-fast responses** (100-150ms) with smart caching
- ✏️ **Advanced spell checking** with 80+ contraction support
- 🚫 **Kick.com detection** with 70+ obfuscation patterns
- 🧠 **5-hook architecture** for seamless feature coordination
- 🔒 **Complete privacy** - everything runs locally

The system combines ChromaDB vector search with Ollama's Gemma 3 12B model for contextually relevant, high-quality suggestions.

## Quick Start (5 minutes)

### What You'll Need
- Mac, Linux, or Windows with 16GB+ RAM
- Node.js 18+ and Python 3.8+
- Ollama installed (https://ollama.ai)
- About 15GB free disk space (for models)
- Basic terminal/command line experience

### Step 1: Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd ai-train-llm

# Install Node dependencies
npm install

# Setup Python environment
cd python
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..

# Setup environment variables
cp .env.example .env.local
# Edit .env.local and set:
# OLLAMA_PATH_API=http://127.0.0.1:11434/api
```

### Step 2: Prepare AI Models
```bash
# Pull the Gemma 3 model (12GB download)
ollama pull gemma3:12b

# Initialize vector database
cd python/vector_db
python setup_chromadb.py
cd ../..
```

### Step 3: Start All Services

**Easy Method (Recommended):**
```bash
# Start all backend services with one command
./start_all_servers.sh

# In another terminal, start the web app
npm run dev
```

**Manual Method (if script fails):**

Open 3 terminals:

**Terminal 1 - Ollama:**
```bash
ollama serve
```

**Terminal 2 - Python API:**
```bash
cd python && python api/api_server.py
```

**Terminal 3 - Web App:**
```bash
npm run dev
```

### Step 4: Verify Everything is Running

1. Open **http://localhost:3000** - You should see the bio form
2. Check **http://localhost:8001/docs** - API documentation
3. Type 5+ words in the bio field and wait 1.5 seconds for suggestions

If you see AI suggestions appearing, everything is working! 🎉

## How Does It Work?

### The Simple Explanation

When you type in the text box, three smart systems work together:

1. **Vector Search (ChromaDB)** 🧠
   - Indexes 5000+ real bio examples
   - Finds semantically similar contexts
   - Provides exact and partial matches

2. **LLM Generation (Ollama)** ✍️
   - Uses Gemma 3 12B model locally
   - Generates contextually relevant completions
   - Filters for quality (8+ word suggestions)

3. **Smart Features** 🛡️
   - Spell checking with typo-js and Hunspell
   - Kick.com detection (70+ patterns)
   - Text feature coordinator prevents conflicts

### The Technical Stack

- **Frontend**: Next.js 15.3.3 + React 19 + TypeScript
- **Styling**: Tailwind CSS v4 with PostCSS
- **AI Engine**: Ollama (Gemma 3 12B) for local inference
- **Python Backend**: FastAPI servers on ports 8001 & 8002
- **Vector Database**: ChromaDB with persistent storage
- **Hook Architecture**: 5 sophisticated React hooks for features
- **Performance**: 100-150ms responses with smart caching

## Using the App

### Writing Your Bio

1. **Enter Your Name**
   - Simple text field
   - Required for submission

2. **Start Typing Your Bio**
   - Begin with something like "I am..." or "We are looking for..."
   - After typing ~5 words, you'll see AI suggestions appear

3. **Accept Suggestions**
   - When you see a gray suggestion you like, press `Tab` to accept it
   - Keep typing to see new suggestions
   - Suggestions appear after a brief pause in typing

### Features While Typing

#### 🎯 Smart Autocomplete
- Waits for natural pauses in your typing
- Shows contextually relevant suggestions
- Press `Tab` to accept, or just keep typing to ignore

#### ✏️ Spell Check
- Underlines misspelled words in red
- Click on any underlined word to see suggestions
- Add words to your personal dictionary

#### 🚫 Kick Detection
- Automatically detects attempts to include "kick.com" links
- Shows warnings for various obfuscation attempts (k1ck, k.i.c.k, etc.)
- Protects the community from spam

### Example Usage

Let's say you start typing:
```
"We are a fun couple looking for"
```

The AI might suggest:
```
"like-minded friends to explore new adventures with"
```

Press `Tab` to accept, and continue typing or wait for more suggestions!

## Customizing Your Experience

### Adding Your Own Bio Examples

The AI learns from examples. To add your own:

1. **Locate the bio data file:**
   ```
   data/bio.json
   ```

2. **Add new examples to the JSON array:**
   ```json
   [
     "Existing bio example...",
     "Your new bio example here",
     "Another example with different style"
   ]
   ```

3. **Rebuild the vector database:**
   ```bash
   cd python/vector_db
   python setup_chromadb.py
   cd ../..
   ```

4. **Restart the API server** for changes to take effect

### Adjusting AI Behavior

You can tweak how suggestions work:

1. **Suggestion Timing**: Edit `hooks/useFormAutocomplete.tsx`
   ```typescript
   const debounceDelay = 1500; // Change from 1.5 to 2 seconds
   ```

2. **Minimum Words**: The AI waits for 5 words by default
   ```typescript
   const minWordsRequired = 5; // Change to 3 for earlier suggestions
   ```

3. **Temperature** (Creativity level): In `actions/ai-text.ts`
   ```typescript
   temperature: 0.85, // Lower = more predictable, Higher = more creative
   ```

### Training Custom Models

For advanced users who want to train their own AI:

1. **Prepare training data** (minimum 100 bio examples)
2. **Run the training script:**
   ```bash
   cd python/mlx_training
   python train_simple.py
   ```
3. See `training_llm_local.md` for detailed instructions

## Understanding the Features

### Why Python Backend?

The Python backend serves as the "brain" of the operation:
- **Vector Search**: Uses ChromaDB to find similar text patterns
- **Model Management**: Handles multiple AI models efficiently
- **Performance**: Processes AI requests without blocking the web interface
- **Flexibility**: Easy to add new AI models or features

### The Hybrid Approach

The app uses two methods to generate suggestions:

1. **Exact Matches**: Finds real bio segments similar to your input
2. **AI Generation**: Creates new, unique suggestions

This combination ensures suggestions are both relevant and creative.

### Content Moderation

The kick detection system protects against:
- Direct spam links (kick.com)
- Obfuscated versions (k1ck, k-i-c-k, etc.)
- Hidden promotional content

While allowing legitimate uses of the word "kick" in normal context.

## Privacy & Safety

### Your Privacy

- ✅ **Everything runs locally** - No data sent to cloud services
- ✅ **No user tracking** - We don't collect any analytics
- ✅ **No bio storage** - Your typed content isn't saved anywhere
- ✅ **Open source** - You can verify the code yourself

### Community Safety

- 🛡️ **Spam protection** built-in
- 🔍 **Content filtering** for inappropriate suggestions
- 👥 **Designed for** adult lifestyle community standards

### Best Practices

1. Don't share personal identifying information in bios
2. Be authentic but protective of your privacy
3. Report any inappropriate AI suggestions you encounter

## Common Questions

### Q: Why do I need three terminal windows?
**A:** Each service runs independently:
- Ollama = The AI brain
- Python API = The coordinator
- Next.js = The web interface

### Q: Can I use this offline?
**A:** Yes! Everything runs locally. No internet needed after initial setup.

### Q: How do I make suggestions appear faster/slower?
**A:** Edit the debounce delay in the code (see Customizing section).

### Q: Is my data private?
**A:** Absolutely. All processing happens on your computer. Nothing is sent to external servers.

### Q: Can I use this for other communities?
**A:** Yes! Replace the bio examples and retrain the models for any community or purpose.

### Q: Why does it block "kick"?
**A:** It detects spam attempts to promote kick.com streaming platform, which violates many community guidelines.

## Troubleshooting

### "API Server Connection Refused"

The Python server isn't running. Fix:
```bash
# In a new terminal:
./start_api_server.sh
```

### "Ollama Not Found"

Ollama isn't installed or running. Fix:
```bash
# Install Ollama first, then:
ollama serve
```

### No Suggestions Appearing

1. Check all three services are running
2. Type at least 5 words
3. Wait 1.5 seconds after stopping
4. Check browser console for errors

### Suggestions Are Poor Quality

1. Add more bio examples to training data
2. Retrain the vector database
3. Adjust temperature settings

### Memory/Performance Issues

For systems with less RAM:
```python
# In python/api/api_server.py
MAX_CONTEXT_LENGTH = 3  # Reduce from 5
CACHE_SIZE = 100  # Reduce from 1000
```

### Getting More Help

1. Check `app_docs/how_to_use.md` for technical details
2. Review existing GitHub issues
3. Ask in the community Discord/Forum

## Next Steps

Now that you're up and running:

1. **Experiment** with different bio starts to see various suggestions
2. **Customize** by adding your own bio examples
3. **Share** feedback to help improve the AI
4. **Explore** advanced features in the technical documentation

Remember: This tool is designed to help you express yourself authentically while maintaining community standards. Have fun with it!

---

*Built with advanced AI technology for the swinger community*