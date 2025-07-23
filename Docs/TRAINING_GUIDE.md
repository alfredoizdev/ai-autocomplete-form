# Training Your Own Model

Train a custom AI model to get faster, more personalized bio suggestions.

## 🎯 Why Train a Model?

- **Speed**: 50-100ms vs 200-500ms responses
- **Consistency**: Learns your specific style
- **Privacy**: Everything stays on your machine
- **Cost**: No API fees after training

## 📋 Requirements

- **Hardware**: Apple Silicon Mac (M1/M2/M3) with 16GB+ RAM
- **Time**: 2-4 hours for training
- **Space**: 20GB free disk space

## 🚀 Quick Training (Recommended)

### Step 1: Prepare High-Quality Data

```bash
# Prepare grammar-filtered high-quality data
./prepare_hq_data_fast.sh
```
This filters the LookingFor dataset for grammatically correct examples:
- Reduces ~15k examples to ~4.5k high-quality ones
- Ensures proper sentence structure and punctuation
- Creates natural split points for better completions
- Takes ~2-3 minutes to process

### Step 2: Start Training
```bash
# Train with optimized parameters
./start_training_mlx_community.sh
```
This trains a HIGH-QUALITY Llama-3.2-3B model:
- **Learning rate**: 1e-5 (optimized for quality)
- **Iterations**: 2000 (thorough training)
- **Layers**: 24 (increased from 16)
- **Time**: ~3-4 hours on Apple Silicon
- **Output**: `models/lookingfor-llama3-3b-hq-lora/`

### Step 3: Use Your Model
```bash
./start_trained.sh
npm run dev
```

## 📊 Understanding the Process

### Training Data Format
The HIGH-QUALITY dataset format ensures superior results:
- **Grammar filtering**: Only grammatically correct sentences included
- **One sentence per example**: Each bio sentence becomes a separate training pair
- **Smart splitting**: Natural breaks at conjunctions, prepositions, key phrases
- **Quality filtering**: Minimum 8 words, maximum 500 words total
- **Proper punctuation**: Prompts end without punctuation, completions end with punctuation
- **Dataset size**: ~4.5k examples (filtered from 15k+ for quality)

### What Happens During Training

1. **Data Preparation**
   - Splits bios into prompt → completion pairs
   - Applies grammar filtering using language_tool_python
   - Each training example is exactly ONE complete sentence
   - Minimum 8 words per sentence requirement
   - Creates natural breaking points at conjunctions, prepositions
   - Outputs to `python/mlx_training/lookingfor_hq/`

2. **Model Training**
   - Base models:
     - Llama-3.2-3B-Instruct-4bit (recommended for quality)
     - Llama-3.2-1B-Instruct-4bit (faster alternative)
   - Applies LoRA (efficient fine-tuning)
   - Saves checkpoints every 100 steps
   - Final models saved to:
     - `models/lookingfor-llama3-3b-hq-lora/` (HIGH-QUALITY 3B, 2000 iterations)
     - `models/lookingfor-llama3-3b-lora/` (Standard 3B, 1500 iterations)
     - `models/lookingfor-llama3-lora/` (1B LookingFor dataset)
     - `models/bio-sentence-llama3-lora/` (bio.json dataset)

3. **Progress Monitoring**
   ```
   # Example from 3B model training:
   Iteration 100: Train loss 1.605, Learning rate 5.00e-05
   Iteration 200: Train loss 1.424, Val loss 1.512
   ```
   - Loss should decrease (lower is better)
   - HIGH-QUALITY 3B model: 2000 iterations (best quality)
   - Standard 3B model: 1500 iterations
   - 1B model: 1000 iterations
   - Validation loss of ~1.5 indicates good convergence

## ⚙️ Customization

### Adjusting Training Parameters

Edit `start_training_mlx_community.sh` to customize:
```bash
LEARNING_RATE=1e-5      # Optimized for quality (was 5e-5)
BATCH_SIZE=4            # Lower = less memory
NUM_ITERATIONS=2000     # More = better quality (was 1500)
NUM_LAYERS=24           # Increased from 16 for better learning
```

### Memory Settings

For different RAM sizes:
```bash
# 16GB RAM
BATCH_SIZE=2
LORA_RANK=8

# 32GB+ RAM  
BATCH_SIZE=4
LORA_RANK=16
```

## 🔧 Manual Training

For full control:
```bash
cd python/mlx_training
source ../venv/bin/activate

# Prepare high-quality data
python prepare_lookingfor_mlx_fast.py

# Train model with optimized parameters
python -m mlx_lm lora \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --train \
  --data lookingfor_hq \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --iters 2000 \
  --val-batches 50 \
  --save-every 200 \
  --adapter-path "adapters/llama3.2-3b-lookingfor-hq" \
  --num-layers 24 \
  --steps-per-report 10 \
  --steps-per-eval 100
```

## 📈 Training Tips

### For Better Quality
- Use grammar-filtered data with `prepare_hq_data_fast.sh`
- Train for 2000 iterations (as configured in the script)
- Use optimized learning rate of 1e-5
- Increase num_layers to 24 for better learning capacity
- Ensure examples are grammatically correct before training

### For Faster Training
- Reduce batch size to 2
- Train for fewer iterations (500)
- Use smaller base model

### Monitoring Training
```bash
# Watch checkpoint creation
watch -n 10 ls -la python/mlx_training/adapters/llama3.2-3b-lookingfor-hq/

# Check GPU usage (Activity Monitor on Mac)
# MLX automatically uses Apple Silicon GPU

# View training progress
tail -f python/mlx_training/training.log
```

## 🎯 Testing Your Model

After training:
```bash
# Quick test
curl -X POST http://localhost:8003/api/autocomplete/mlx \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I am a fun loving couple who"}'
```

Expected response times:
```json
{
  "completion": "enjoys meeting new people and exploring new experiences.",
  "elapsed_ms": 123.45,
  "model_name": "llama3.2-mlx-finetuned"
}
```
- **3B Model**: 100-150ms per completion
- **1B Model**: 50-100ms per completion
```

## 🚨 Common Issues

### "Out of Memory"
```bash
# Reduce batch size in start_training.sh
BATCH_SIZE=1
```

### Training Too Slow
- Normal: ~2 seconds per iteration
- Close other applications
- Ensure Mac is plugged in

### Poor Results
- Check training data quality
- Train for more iterations
- Try different learning rate

## 💾 Managing Models

### Backup Your Model
```bash
cp -r models/bio-llama3-lora models/bio-llama3-lora-backup
```

### Compare Models
Train multiple versions and A/B test:
```bash
# Version 1
./start_training.sh

# Backup
mv models/bio-llama3-lora models/v1-1000-iters

# Version 2 with different settings
# Edit start_training.sh, then:
./start_training.sh
```

## 🎓 Advanced Topics

### Using Different Base Models
```bash
# HIGH-QUALITY model (default, best results)
MODEL_NAME="mlx-community/Llama-3.2-3B-Instruct-4bit"
NUM_ITERATIONS=2000
LEARNING_RATE=1e-5

# Standard quality model
MODEL_NAME="mlx-community/Llama-3.2-3B-Instruct-4bit"
NUM_ITERATIONS=1500
LEARNING_RATE=5e-5

# Faster training and inference
MODEL_NAME="mlx-community/Llama-3.2-1B-Instruct-4bit"
NUM_ITERATIONS=1000
LEARNING_RATE=5e-5
```

### Resume Training
```bash
# In start_training_mlx_community.sh, add:
--resume-adapter-path adapters/llama3.2-3b-lookingfor-hq/0000500_adapters.safetensors
```

## 📚 Next Steps

1. **Experiment**: Try different training parameters
2. **Iterate**: Add more training data
3. **Deploy**: Use PM2 for production
4. **Share**: Export your model for others

Need help? Check the [Troubleshooting Guide](./TROUBLESHOOTING.md) or open an issue!