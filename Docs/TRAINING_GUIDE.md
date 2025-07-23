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

### Step 1: Prepare Data
```bash
./prepare_training_data.sh
```
This converts bio examples into training format (~5 minutes).

### Step 2: Start Training
```bash
./start_training.sh
```
This trains a Llama-3.2 model with your data (2-4 hours).

### Step 3: Use Your Model
```bash
./start_trained.sh
npm run dev
```

## 📊 Understanding the Process

### What Happens During Training

1. **Data Preparation**
   - Splits bios into prompt → completion pairs
   - Creates natural breaking points
   - Outputs to `python/mlx_training/bio_mlx_improved/`

2. **Model Training**
   - Uses Llama-3.2-1B as base model
   - Applies LoRA (efficient fine-tuning)
   - Saves checkpoints every 100 steps
   - Final model in `models/bio-llama3-lora/`

3. **Progress Monitoring**
   ```
   Iteration 100: Train loss 2.145, Learning rate 5.00e-05
   Iteration 200: Train loss 1.823, Learning rate 5.00e-05
   ```
   - Loss should decrease (lower is better)
   - Training completes at iteration 1000

## ⚙️ Customization

### Adjusting Training Parameters

Edit `start_training.sh` to customize:
```bash
LEARNING_RATE=5e-5      # Lower = more stable
BATCH_SIZE=4            # Lower = less memory
NUM_ITERATIONS=1000     # More = better quality
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
cd python
source venv/bin/activate

# Prepare data
python mlx_training/prepare_bio_mlx_improved.py

# Train model
python -m mlx_lm.lora \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --train \
  --data ../python/mlx_training/bio_mlx_improved \
  --batch-size 4 \
  --iters 1000 \
  --adapter-path ../models/my-custom-model
```

## 📈 Training Tips

### For Better Quality
- Add more diverse bio examples to `data/bio.json`
- Train for more iterations (1500-2000)
- Use lower learning rate (2e-5)

### For Faster Training
- Reduce batch size to 2
- Train for fewer iterations (500)
- Use smaller base model

### Monitoring Training
```bash
# Watch checkpoint creation
watch -n 10 ls -la models/bio-llama3-lora-new/

# Check GPU usage (Activity Monitor on Mac)
# MLX automatically uses Apple Silicon GPU
```

## 🎯 Testing Your Model

After training:
```bash
# Quick test
curl -X POST http://localhost:8003/api/autocomplete/mlx \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I am a fun loving couple who"}'
```

Expected response:
```json
{
  "completion": "enjoys meeting new people and exploring...",
  "elapsed_ms": 73.45,
  "model_name": "llama3.2-mlx-finetuned"
}
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
# Larger model (better quality, slower)
MODEL_NAME="mlx-community/Llama-3.2-3B-Instruct-4bit"

# Smaller model (faster, less quality)
MODEL_NAME="mlx-community/Qwen2.5-1.5B-Instruct-4bit"
```

### Resume Training
```bash
# In start_training.sh, add:
--resume-adapter-path models/bio-llama3-lora/0000500_adapters.safetensors
```

## 📚 Next Steps

1. **Experiment**: Try different training parameters
2. **Iterate**: Add more training data
3. **Deploy**: Use PM2 for production
4. **Share**: Export your model for others

Need help? Check the [Troubleshooting Guide](./TROUBLESHOOTING.md) or open an issue!