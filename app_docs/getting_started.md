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

The AI Bio Generator is a smart text completion tool designed specifically for the adult lifestyle community. It helps you write compelling personal bios by:

- 🤖 **Suggesting completions** as you type
- ✨ **Understanding context** from the lifestyle community
- 🚫 **Detecting inappropriate content** (like promotional links)
- ✏️ **Checking spelling** in real-time
- 🔒 **Keeping everything private** on your local machine

Think of it as an intelligent writing assistant that understands the nuances of lifestyle community language and helps you express yourself authentically.

## Quick Start (5 minutes)

### What You'll Need
- A Mac or PC with at least 16GB RAM
- Basic comfort with Terminal/Command Prompt
- About 10GB free disk space

### Step 1: Get the Code
```bash
# Clone the repository
git clone https://github.com/your-repo/ai-train-llm.git
cd ai-train-llm

# Install dependencies
npm install
```

### Step 2: Start the AI Services
Open **three separate terminal windows**:

**Terminal 1 - Start Ollama (AI Engine):**
```bash
# If you haven't installed Ollama yet:
# Mac: brew install ollama
# Windows/Linux: See https://ollama.ai

ollama serve
```

**Terminal 2 - Start Python API:**
```bash
./start_api_server.sh
```

**Terminal 3 - Start the Web App:**
```bash
npm run dev
```

### Step 3: Open Your Browser
Navigate to: **http://localhost:3000**

That's it! You should see the bio generator interface. 🎉

## How Does It Work?

### The Simple Explanation

When you type in the text box, three smart systems work together:

1. **Memory Bank** 🧠
   - Stores thousands of example bios
   - Finds similar phrases to what you're typing
   - Like having a friend who remembers every good bio they've seen

2. **AI Writer** ✍️
   - Takes inspiration from the memory bank
   - Generates new, creative suggestions
   - Understands lifestyle community language and tone

3. **Safety Guard** 🛡️
   - Checks for spam and promotional content
   - Ensures suggestions are appropriate
   - Keeps the community safe from bad actors

### The Technical Stack (for the curious)

- **Frontend**: Next.js 15 with React 19 (modern web framework)
- **AI Engine**: Ollama running Gemma 3 12B model (local AI, no cloud)
- **Python Backend**: FastAPI server handling AI orchestration
- **Vector Database**: ChromaDB for intelligent text matching
- **Custom Models**: Fine-tuned GPT-2 for lifestyle-specific suggestions

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
   data/Bios.ts
   ```

2. **Add new examples to the array:**
   ```typescript
   export const Bios = [
     // Existing bios...
     "Your new bio example here",
     "Another example with different style",
   ];
   ```

3. **Update the vector database:**
   ```bash
   cd python/vector_db
   python setup_chromadb.py --reset
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

*Built with ❤️ for the lifestyle community by Swing.com*