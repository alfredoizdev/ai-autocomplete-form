# Model Training Guide

## Overview

This guide explains how to train custom models for the AI Bio Autocomplete system. While the default setup uses Ollama's Gemma 3 12B model, you can train specialized models for better performance and domain-specific understanding.

## Current Architecture

The system supports two approaches:

1. **Default**: Ollama Gemma 3 12B (no training needed)
2. **Fine-tuned**: Custom GPT-2/DistilGPT2 models trained on bio data

## Quick Start: Using Pre-trained Models

The easiest option is to use the existing fine-tuned models:

```bash
# Start the trained model server
cd python
python -m uvicorn api.trained_model_server:app --port 8002

# Enable in .env.local
NEXT_PUBLIC_USE_FINETUNED_MODEL=true
```

Available pre-trained models:
- `bio_distilgpt2_finetuned` - Lightweight, fast (default)
- `bio_gpt2_improved` - Larger, more accurate
- `bio_gpt2_finetuned` - Standard GPT-2

## Training Your Own Model

### Prerequisites

- Python 3.8+
- 16GB+ RAM (32GB recommended)
- CUDA GPU (optional but 10x faster)
- ~5GB disk space per model

### Step 1: Prepare Training Data

Your bio data should be in JSON format:

```json
[
  "Professional couple seeking like-minded friends...",
  "We are new to the lifestyle and looking to explore...",
  "Fun-loving pair interested in meeting others..."
]
```

Place your data in `data/bio.json` (minimum 100 examples, 1000+ recommended).

### Step 2: Setup Training Environment

```bash
cd python/mlx_training

# Create virtual environment
python3 -m venv training_env
source training_env/bin/activate  # Windows: training_env\Scripts\activate

# Install dependencies
pip install torch transformers datasets accelerate
```

### Step 3: Train DistilGPT2 (Recommended)

DistilGPT2 offers the best balance of speed and quality:

```bash
python train_distilgpt2.py
```

Training parameters (edit in script):
```python
training_args = TrainingArguments(
    output_dir="./bio_distilgpt2_finetuned",
    num_train_epochs=3,              # Increase for better quality
    per_device_train_batch_size=4,   # Decrease if OOM
    warmup_steps=100,
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
)
```

### Step 4: Train GPT-2 with LoRA (Advanced)

For better quality with memory efficiency:

```bash
python train_bio_improved.py
```

This uses LoRA (Low-Rank Adaptation) to train only small adapter layers.

### Step 5: Test Your Model

```python
from transformers import pipeline

# Load your trained model
generator = pipeline(
    'text-generation', 
    model='./bio_distilgpt2_finetuned',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Test generation
result = generator(
    "I am looking for",
    max_new_tokens=30,
    temperature=0.8,
    do_sample=True
)
print(result[0]['generated_text'])
```

### Step 6: Deploy Your Model

1. **Update the model path** in `api/trained_model_server.py`:
   ```python
   model_path = "mlx_training/your_model_name"
   ```

2. **Restart the server**:
   ```bash
   cd python
   python -m uvicorn api.trained_model_server:app --port 8002
   ```

3. **Test the endpoint**:
   ```bash
   curl -X POST http://localhost:8002/api/autocomplete/trained \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Looking for"}'
   ```

## Training Tips

### Data Quality
- Remove duplicates and low-quality bios
- Ensure diverse writing styles
- Balance different bio lengths
- Clean inappropriate content

### Hyperparameter Tuning
- **Epochs**: 3-5 usually sufficient
- **Batch Size**: 4-8 for DistilGPT2, 2-4 for GPT-2
- **Learning Rate**: 5e-5 is a good starting point
- **Max Length**: 128 tokens covers most bios

### Performance Optimization
- Use GPU if available (10x faster)
- Enable mixed precision training
- Use gradient accumulation for larger effective batch size
- Save checkpoints to resume training

### Common Issues

**Out of Memory:**
```python
# Reduce batch size
per_device_train_batch_size=2

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use 8-bit training
load_in_8bit=True
```

**Poor Generation Quality:**
- Train for more epochs
- Increase training data
- Adjust temperature (0.7-0.9)
- Try different model architectures

**Slow Training:**
- Use DistilGPT2 instead of GPT-2
- Enable GPU acceleration
- Reduce max sequence length
- Use cached datasets

## Evaluating Your Model

### Metrics to Track
- **Perplexity**: Lower is better (aim for <20)
- **Generation Speed**: Target <150ms
- **Memory Usage**: Should fit in available RAM
- **Quality**: Manual review of generations

### A/B Testing
Compare your model against the default:

```python
# In api_server.py, add model selection
@app.post("/api/autocomplete/compare")
async def compare_models(request: AutocompleteRequest):
    # Get suggestions from both models
    default = await get_default_suggestion(request.prompt)
    custom = await get_custom_suggestion(request.prompt)
    
    return {
        "default": default,
        "custom": custom,
        "prompt": request.prompt
    }
```

## Advanced Training

### Multi-GPU Training
```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    train_distilgpt2.py
```

### Custom Tokenization
```python
# Add special tokens for community terms
tokenizer.add_special_tokens({
    'additional_special_tokens': [
        '[COUPLE]', '[SINGLE]', '[LIFESTYLE]'
    ]
})
```

### Domain Adaptation
Fine-tune on progressively more specific data:
1. General text → Bio text → Community-specific bios

## Troubleshooting

### Model Not Generating
- Check model loaded correctly
- Verify tokenizer compatibility
- Ensure prompt format matches training

### Server Not Starting
- Check port 8002 availability
- Verify model path exists
- Review server logs for errors

### Poor Autocomplete Quality
- Ensure enough training data (1000+ examples)
- Check data preprocessing
- Adjust generation parameters

## Next Steps

1. **Experiment** with different model architectures
2. **Collect** more domain-specific training data
3. **Monitor** user feedback and model performance
4. **Iterate** on training based on real usage

---

*For API integration details, see [API Server Guide](API_SERVER_GUIDE.md)*