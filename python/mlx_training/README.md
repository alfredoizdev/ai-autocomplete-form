# MLX Training for Bio Autocomplete

## Overview

This module implements Phase 2 of the bio autocomplete enhancement project. It uses Apple's MLX framework to fine-tune a language model (Gemma 2B) on your bio data, optimized for M1 Mac with 32GB RAM.

## Current Status

✅ **Completed:**
- MLX environment setup
- Training data preparation (4,599 training examples)
- Training pipeline configuration
- Model conversion pipeline
- Ollama integration setup

⏳ **Pending:**
- Actual model download (requires Hugging Face license acceptance)
- Model fine-tuning execution
- Integration with existing autocomplete system

## Quick Start

### 1. Prepare Training Data
```bash
python prepare_mlx_data.py
```
- Converts 514 bios into 5,110 training examples
- Creates train/validation split (90/10)
- Saves datasets in `bio_dataset/`

### 2. Setup Model (First Time)
```bash
# Accept Gemma license at: https://huggingface.co/google/gemma-2b
# Login to Hugging Face
huggingface-cli login

# Download model
python download_model.py
```

### 3. Train Model
```bash
python train_mlx.py
```
- Uses LoRA for memory-efficient training
- Configured for 1 epoch (~2-3 hours on M1 Max)
- Saves adapters to `adapters/bio_lora/`

### 4. Convert for Ollama
```bash
python convert_to_gguf.py
```
- Creates Modelfile for Ollama
- Follow prompts to create `bio-autocomplete` model

### 5. Test Model
```bash
ollama run bio-autocomplete "We are a couple who"
```

## Training Configuration

Optimized for M1 Mac with 32GB RAM:
- Model: Gemma 2B (smaller than originally planned 9B)
- LoRA rank: 8 (memory efficient)
- Batch size: 1 with gradient accumulation
- Max sequence length: 256 tokens
- Target modules: q_proj, v_proj (attention only)

## Integration with Existing System

The trained model will work alongside the existing:
- Vector search API (port 8001) - <100ms responses
- Original Gemma 3:12B fallback
- Spell check and auto-capitalization

No existing functionality will be disrupted.

## Files Created

```
mlx_training/
├── README.md                 # This file
├── prepare_mlx_data.py      # Data preparation
├── download_model.py        # Model download helper
├── train_mlx.py            # Training script
├── convert_to_gguf.py      # Ollama conversion
├── bio_dataset/            # Prepared datasets
│   ├── train/
│   ├── validation/
│   └── dataset_stats.json
├── models/                 # Model storage
│   └── model_config.json
└── adapters/              # LoRA adapters
    └── bio_lora/
        ├── training_config.json
        ├── training_log.json
        ├── conversion_meta.json
        └── Modelfile
```

## Next Steps

1. Accept Gemma license and download model
2. Run actual training (2-3 hours)
3. Create Ollama model from fine-tuned weights
4. Update `ai-text.ts` to include new model option
5. A/B test against current implementation

## Performance Expectations

- Training time: 2-3 hours on M1 Max
- Memory usage: ~15-20GB during training
- Model size: ~1.6GB (Gemma 2B base + LoRA adapters)
- Inference speed: Similar to current Gemma models

## Troubleshooting

### Out of Memory
- Reduce batch_size in train_mlx.py
- Lower max_seq_length
- Use fewer LoRA target modules

### Slow Training
- Normal for first epoch (2-3 hours expected)
- MLX optimizes for Apple Silicon
- Monitor Activity Monitor for GPU usage

### Model Not Found
- Ensure Gemma license accepted
- Run `huggingface-cli login`
- Check internet connection