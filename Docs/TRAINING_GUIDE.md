# Training Your Own AI Model

Want the app to write in a specific style? This guide shows you how to create your own custom AI model!

## 🤔 What is Model Training?

Think of it like teaching someone your writing style:
- You show them thousands of examples
- They learn the patterns
- Eventually, they can write like those examples

In this case, you're teaching an AI to complete bio sentences in a specific way.

## 🎯 Why Train Your Own Model?

### Without Training (Hybrid Mode)
- Uses a general-purpose AI (like a jack-of-all-trades writer)
- Good quality but not specialized
- Slightly slower (searches database first)

### With Training (Trained Mode) 
- AI becomes a specialist in bio writing
- Faster responses (no database search needed)
- Consistent style based on your training data
- Everything runs locally - total privacy

## 📋 Before You Start

### Requirements
- **Computer**: Mac with Apple Silicon chip (M1, M2, or M3)
- **Memory**: 16GB RAM minimum (32GB recommended)
- **Storage**: 20GB free space
- **Time**: 3-4 hours (mostly waiting)
- **Patience**: The computer does the work, but it takes time!

## 🚀 The Simple 3-Step Process

### Step 1: Prepare the Training Data (5 minutes)

```bash
./prepare_hq_data_fast.sh
```

**What this does:**
- Takes 15,000+ bio examples
- Filters out poorly written ones
- Keeps only grammatically correct sentences
- Splits them into "prompt → completion" pairs
- Results in ~4,500 high-quality training examples

**You'll see:**
```
Processing bio data...
Filtered 15,234 examples down to 4,521 high-quality examples
Training data ready!
```

### Step 2: Train Your Model (3-4 hours)

```bash
./start_training_mlx_community.sh
```

**What this does:**
- Downloads a base AI model (Llama 3.2)
- Shows it your prepared bio examples
- Teaches it to complete sentences in that style
- Saves checkpoints every 100 steps (in case something goes wrong)
- Creates your custom model in `models/` folder

**You'll see progress like:**
```
Iteration 100: Train loss 1.605, Learning rate 1.00e-05
Iteration 200: Train loss 1.424, Val loss 1.512
...
Iteration 2000: Train loss 0.832, Val loss 1.201
Training complete! Model saved to models/lookingfor-llama3-3b-hq-lora/
```

**Tips while waiting:**
- Don't close the terminal!
- Your Mac might get warm - this is normal
- You can use your computer for other things
- Check progress every 30 minutes or so

### Step 3: Use Your Trained Model

```bash
# Terminal 1:
./start_trained.sh

# Terminal 2:
npm run dev
```

Visit http://localhost:3000 and start typing! Your custom AI is now powering the suggestions.

## 📊 Understanding What's Happening

### How the Training Data Works

Imagine teaching someone to complete sentences. You show them:
- **Start**: "I am looking for couples who"
- **Completion**: "enjoy good conversation and laughter."

The AI learns from 4,500+ examples like this, understanding patterns in how bios are written.

### What Makes Good Training Data?
- ✅ Complete, grammatically correct sentences
- ✅ Natural language that flows well
- ✅ Varied examples (not all the same)
- ❌ Typos, incomplete thoughts
- ❌ Inappropriate content
- ❌ Sentences that are too short (<8 words)

### Reading the Training Progress

When training runs, you'll see numbers like:
```
Iteration 500: Train loss 1.234, Val loss 1.456
```

**What these mean:**
- **Iteration**: How many times it's practiced (out of 2000)
- **Train loss**: How well it's learning (lower = better)
- **Val loss**: How well it works on new examples (lower = better)

**Good signs:**
- Both numbers going down over time
- Val loss around 1.0-1.5 at the end
- Steady progress without big jumps

### Where Your Model is Saved

After training completes, your model is saved in:
```
models/
└── lookingfor-llama3-3b-hq-lora/
    ├── adapter_config.json     (model settings)
    ├── adapters.safetensors    (the trained weights)
    └── training_results.json   (performance metrics)
```

## ⚙️ Customization (Advanced)

### If You Have Less Memory (16GB RAM)

Edit `start_training_mlx_community.sh` and change:
```bash
BATCH_SIZE=2    # Was 4
```
This will take longer but use less memory.

### Want Faster Training?

Change these settings for speed over quality:
```bash
NUM_ITERATIONS=1000    # Was 2000 (half the time)
```

### Using Your Own Training Data

If you have your own bio examples:
1. Save them as a CSV file in `data/` folder
2. Make sure first column contains the bio text
3. Run the data preparation script
4. Follow the training steps above

For detailed customization, see the [Development Guide](./DEVELOPMENT_GUIDE.md).

## 💡 Training Best Practices

### Do's ✅
- **Keep your Mac plugged in** - Training uses lots of power
- **Close heavy apps** - Free up memory for training
- **Check progress every hour** - Make sure loss is decreasing
- **Save your work** - The script auto-saves checkpoints

### Don'ts ❌
- **Don't close the terminal** - This stops training
- **Don't put Mac to sleep** - Disable sleep in System Settings
- **Don't panic if it's slow** - First 100 iterations are always slowest
- **Don't stop early** - Let it complete all 2000 iterations

## 🧪 Testing Your Model

After training completes, test it quickly:

1. Make sure trained mode is running:
   ```bash
   ./start_trained.sh
   ```

2. Open a new terminal and test:
   ```bash
   curl -X POST http://localhost:8003/api/autocomplete/mlx \
     -H "Content-Type: application/json" \
     -d '{"prompt": "We are looking for"}'
   ```

3. You should see a bio completion in ~100ms!

## 🚨 Troubleshooting Training

**"Out of memory" error?**
- See memory settings in Customization section above
- Close other apps
- Restart your Mac and try again

**Training seems frozen?**
- Check Activity Monitor - if Python is using CPU, it's working
- First 100 iterations are slowest
- Some iterations take longer than others (normal)

**Poor quality results?**
- Make sure you used the HIGH-QUALITY data script
- Let training complete all 2000 iterations
- Check the [Troubleshooting Guide](./TROUBLESHOOTING.md)

## 🎉 Congratulations!

If you've made it this far, you've:
- Prepared professional-quality training data
- Trained your own AI model
- Created a custom bio completion system

This is advanced AI work that most developers never attempt. Well done!

## 📚 What's Next?

- **Use your model**: Follow the [Running Guide](./RUNNING_THE_APP.md) 
- **Train with custom data**: Add your own bio examples
- **Share your experience**: Let us know how it went!

---

*Remember: Training takes time, but the result is your very own AI assistant. Be patient and enjoy watching your computer learn!*