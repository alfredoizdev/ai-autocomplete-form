# Local LLM Training Guide

This guide will teach you how to train your own custom AI model for bio autocomplete. Even if you've never trained an AI model before, you'll be able to follow along!

## 🎯 Why Train a Custom Model?

### Benefits of Custom Training
- **Faster Response**: 50-100ms vs 200-500ms
- **Better Quality**: Trained specifically on your data
- **Offline Usage**: No internet needed after training
- **Cost Effective**: No API fees

### When to Use Custom Models
- You have specific style requirements
- You need faster response times
- You want complete control over the AI
- You have good training data

## 🖥️ Requirements

### Hardware Requirements
- **Required**: Apple Silicon Mac (M1, M2, M3)
- **RAM**: 32GB recommended (16GB minimum)
- **Storage**: 20GB free space
- **Time**: 2-4 hours for training

### Software Requirements
- All requirements from the basic setup
- MLX framework (included in requirements.txt)

## 📚 Understanding the Training Process

### What is Fine-Tuning?
- Starting with a pre-trained model (Phi-3)
- Teaching it your specific style with your data
- Like teaching someone your writing style

### What is LoRA?
- **L**ow-**R**ank **A**daptation
- A memory-efficient training method
- Trains only a small part of the model
- Perfect for consumer hardware

### The Training Pipeline
```
Raw Bios → Data Preparation → Training → Fine-tuned Model → Deployment
```

## 🚀 Step-by-Step Training Guide

### Step 1: Prepare Your Environment
```bash
# Navigate to the MLX training directory
cd python/mlx_server

# Ensure virtual environment is activated
source ../venv/bin/activate

# Verify MLX is installed
python -c "import mlx; print('MLX is ready!')"
```

### Step 2: Prepare Training Data

The data preparation script converts your bio examples into training format:

```bash
# Go to the mlx_training directory
cd ../mlx_training

# Run the preparation script
python prepare_bio_mlx_improved.py

# You should see:
# ✅ Created 4000 training examples
# ✅ Created 500 validation examples
# ✅ Created 495 test examples
```

**What This Does:**
- Splits each bio into prompt → completion pairs
- Creates natural breaking points
- Ensures high-quality training data

### Step 3: Copy Data to MLX Server
```bash
# Copy prepared data to MLX server directory
cp bio_dataset/*.jsonl ../mlx_server/data/

# Verify files exist
ls ../mlx_server/data/
# Should show: train.jsonl, valid.jsonl, test.jsonl
```

### Step 4: Configure Training

Check the configuration file:
```bash
cd ../mlx_server
cat config.yaml
```

Key settings explained:
```yaml
model: "mlx-community/Phi-3-mini-4k-instruct-4bit"  # Base model
lora:
  rank: 16              # Lower = less memory, Higher = more capacity
training:
  batch_size: 2         # Samples per training step
  num_epochs: 3         # Times through the data
  learning_rate: 5e-5   # How fast to learn
```

### Step 5: Start Training!
```bash
# Run the training script
python train.py
```

**What You'll See:**
```
Checking for model: mlx-community/Phi-3-mini-4k-instruct-4bit
✅ Model is ready
Starting MLX LoRA training...

Training Progress:
--------------------------------------------------
Step 10/1000 | Loss: 2.45 | LR: 5e-5 | Time: 0.5s
Step 20/1000 | Loss: 2.12 | LR: 5e-5 | Time: 0.5s
...
```

**Training Tips:**
- Loss should decrease over time
- Each step takes 0.5-2 seconds
- Total time: 2-4 hours
- Don't close the terminal!

### Step 6: Monitor Training
```bash
# In another terminal, watch the model directory
watch -n 10 ls -la models/bio-phi3-lora/
```

You'll see checkpoint files appearing:
- `checkpoint-100.safetensors`
- `checkpoint-200.safetensors`
- etc.

### Step 7: Test Your Model

After training completes:

```bash
# Start the MLX model server
python mlx_model_server.py
```

Test with curl:
```bash
curl -X POST http://localhost:8003/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I am a fun loving person who"}'
```

## 🎮 Using Your Trained Model

### Step 1: Enable in Frontend
```bash
# Edit .env.local
echo "NEXT_PUBLIC_USE_FINETUNED_MODEL=true" >> .env.local
```

### Step 2: Start All Services
```bash
# Terminal 1: Ollama (still needed as fallback)
ollama serve

# Terminal 2: API Server
cd python && python api/api_server.py

# Terminal 3: MLX Model Server
cd python/mlx_server && python mlx_model_server.py

# Terminal 4: Frontend
npm run dev
```

### Step 3: Verify It's Working
- Open http://localhost:3000
- Type a bio prompt
- Check the browser console for "Using finetuned model"
- Responses should be faster!

## 📊 Understanding Training Metrics

### Loss Values
- **Starting Loss**: 2.5-3.5 (normal)
- **Good Loss**: 1.0-1.5
- **Overfitting**: Loss stops decreasing or increases

### Validation Metrics
- Checked every 50 steps
- Should follow training loss
- Big gap = overfitting

### When to Stop Training
- Loss plateaus
- Validation loss increases
- 1000-2000 steps usually sufficient

## 🔧 Customizing Training

### Adjusting for Your Hardware

**For 16GB RAM:**
```yaml
training:
  batch_size: 1
  gradient_accumulation: 8
lora:
  rank: 8
```

**For 64GB RAM:**
```yaml
training:
  batch_size: 4
  gradient_accumulation: 2
lora:
  rank: 32
```

### Improving Quality

**More Training Data:**
1. Add more bios to `data/bio.json`
2. Re-run data preparation
3. Train longer (more epochs)

**Better Data Quality:**
- Remove duplicates
- Fix spelling errors
- Ensure complete sentences

**Training Strategies:**
```yaml
# For more creativity
training:
  learning_rate: 1e-4  # Higher learning rate

# For more consistency  
training:
  learning_rate: 1e-5  # Lower learning rate
  num_epochs: 5        # More epochs
```

## 🎯 Troubleshooting Training

### "Out of Memory" Error
```bash
# Reduce batch size in config.yaml
batch_size: 1

# Or reduce sequence length
max_seq_length: 128
```

### Training Too Slow
```bash
# Check GPU usage
# MLX automatically uses Apple Silicon GPU

# Reduce logging frequency
logging_steps: 50  # Instead of 10
```

### Poor Model Quality
1. **Check Training Data**:
   ```bash
   # Look at prepared data
   head -20 data/train.jsonl
   ```

2. **Train Longer**:
   ```yaml
   num_epochs: 5  # Instead of 3
   ```

3. **Adjust Learning Rate**:
   ```yaml
   learning_rate: 2e-5  # Try different values
   ```

## 📈 Advanced Training Tips

### 1. Resume Training
```yaml
# In config.yaml
output:
  resume_from_checkpoint: "models/bio-phi3-lora/checkpoint-500"
```

### 2. Multiple Training Runs
```bash
# Save different versions
cp -r models/bio-phi3-lora models/bio-phi3-lora-v1
```

### 3. A/B Testing
Run two model servers on different ports:
```bash
# Model 1
python mlx_model_server.py --port 8003

# Model 2  
python mlx_model_server.py --port 8004 --model models/bio-phi3-lora-v2
```

## 🎓 Understanding the Code

### Data Preparation (`prepare_bio_mlx_improved.py`)
- Splits bios at natural points (conjunctions, prepositions)
- Creates prompt → completion pairs
- Validates data quality

### Training Script (`train.py`)
- Downloads base model if needed
- Applies LoRA adapters
- Saves checkpoints regularly

### Model Server (`mlx_model_server.py`)
- Loads trained model
- Handles generation requests  
- Includes grammar fixing

## 📊 Comparing Models

| Feature | Ollama (Gemma 3) | Fine-tuned (Phi-3) |
|---------|------------------|-------------------|
| Response Time | 200-500ms | 50-100ms |
| Model Size | 12GB | 2GB + adapters |
| Quality | General purpose | Bio-specific |
| Setup Time | 5 minutes | 2-4 hours |

## 🚀 Production Deployment

### Optimizing for Production
1. Use the best checkpoint (not necessarily the last)
2. Quantize further if needed
3. Set up model caching
4. Monitor performance

### Serving Multiple Users
```python
# In mlx_model_server.py
# Adjust max_workers for concurrent requests
app = FastAPI()
# Add caching, rate limiting, etc.
```

## 📚 Next Steps

1. **Experiment with Data**: Try different bio styles
2. **Try Different Models**: Mistral, Qwen, etc.
3. **Optimize Performance**: Caching, batching
4. **Deploy to Cloud**: Use Mac cloud instances

## 🎉 Congratulations!

You've successfully trained your own AI model! This is a significant achievement. Your custom model is now:
- Specialized for bio writing
- Fast and efficient
- Completely under your control

Remember: Good AI models come from good data and patience with training. Keep experimenting!

---

For more help, check out:
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)
- Project GitHub Issues