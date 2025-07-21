# MLX Training Quick Start Guide (2025)

## 🚀 Quick Start Commands

Run these commands in order to train your bio autocomplete model:

```bash
# 1. Install/update dependencies
pip install --upgrade mlx mlx-lm transformers datasets huggingface-hub

# 2. Login to Hugging Face (required for model downloads)
huggingface-cli login

# 3. Run the setup pipeline
cd python/mlx_training
python train_mlx_updated.py

# 4. Download and convert the model (4-bit quantized)
python -m mlx_lm.convert --hf-path mistralai/Mistral-7B-Instruct-v0.2 -q

# 5. Start training
./run_training.sh

# 6. Test the fine-tuned model
python test_finetuned_model.py "We are a fun couple who"
```

## 📊 Training Parameters (Optimized for M1 Max 32GB)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Mistral 7B | Best balance of quality/speed |
| LoRA Rank | 8 | Increase to 16 if memory allows |
| Batch Size | 2 | With gradient accumulation = 8 |
| Learning Rate | 2e-4 | Standard for LoRA fine-tuning |
| Max Sequence | 512 | Good for bio texts |
| Training Time | 3-4 hours | On M1 Max |
| Memory Usage | 20-25GB | Peak during training |

## 🎯 What This Does

1. **Uses existing bio dataset** (4,599 examples)
2. **Fine-tunes with LoRA** for memory efficiency
3. **Creates custom bio autocomplete model**
4. **Saves adapter files** (~200MB)
5. **Provides testing script** for immediate use

## 💡 Memory Optimization Tips

If you run out of memory:

```yaml
# Reduce these in training_config.yaml:
lora_layers: 8  # Instead of 16
batch_size: 1   # Instead of 2
max_seq_length: 256  # Instead of 512
```

## 🔧 Alternative Models (By RAM Usage)

- **16GB RAM**: Use Gemma 2B or Phi-3 Mini
- **32GB RAM**: Mistral 7B (recommended) or Llama 3 8B
- **64GB RAM**: Can handle 13B models comfortably

## 📈 Monitor Training

- Use Activity Monitor to watch GPU usage
- Training logs saved to `adapters/bio_mistral_lora/`
- Optional: Use `--wandb` flag for cloud monitoring

## 🏁 After Training

1. **Test quality**: 
   ```bash
   python test_finetuned_model.py "Looking for"
   ```

2. **Convert for Ollama** (optional):
   ```bash
   python convert_to_gguf.py
   ```

3. **Integrate with API**:
   - Update `api_server.py` to use MLX model
   - Add as additional backend option

## ⚡ Performance Expectations

- **Inference**: 50-100 tokens/second
- **API Response**: <100ms for short completions  
- **Model Size**: ~4GB (quantized) + 200MB adapters

## 🆘 Troubleshooting

**"mlx_lm not found"**
```bash
pip uninstall mlx-lm mlx
pip install --upgrade mlx mlx-lm
```

**"Out of memory"**
- Close other applications
- Reduce training parameters (see above)
- Try 4-bit quantized model

**"Model not found"**
- Ensure HF login: `huggingface-cli whoami`
- Check internet connection
- Try direct download URL

## 📚 Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX-LM GitHub](https://github.com/ml-explore/mlx-lm)
- [MLX Community Models](https://huggingface.co/mlx-community)

Ready to train! The entire process should take 4-5 hours including setup.