# MLX Bio Autocomplete Training

This directory contains the MLX-based training and serving infrastructure for the bio autocomplete model.

## Overview

We use Apple's MLX framework to fine-tune a Phi-3-mini model (3.8B parameters) for bio text completion. The model is optimized for Apple Silicon and provides fast inference on M1/M2/M3 Macs.

## Directory Structure

```
mlx_server/
├── data/                    # Converted training data in MLX format
│   ├── train.jsonl         # 2,922 training samples
│   ├── valid.jsonl         # 365 validation samples
│   └── test.jsonl          # 366 test samples
├── models/                 # Trained model adapters
│   └── bio-phi3-lora/     # LoRA adapter files
├── config.yaml            # Training configuration
├── convert_to_mlx_format.py  # Data conversion script
├── train.py               # Training script
├── mlx_model_server.py    # FastAPI server (port 8003)
├── test_mlx_server.py     # Server testing script
└── requirements.txt       # Python dependencies
```

## Quick Start

### 1. Training the Model

```bash
# From the mlx_server directory
python train.py
```

This will:
- Download the Phi-3-mini-4k-instruct model (first time only)
- Fine-tune it with LoRA on your bio data
- Save checkpoints every 100 steps
- Complete in approximately 20-40 minutes on M1 Max

### 2. Starting the Server

```bash
python mlx_model_server.py
```

The server runs on port 8003 and provides:
- `/api/autocomplete/mlx` - Single completion endpoint
- `/api/autocomplete/mlx/batch` - Batch completions
- `/health` - Health check
- `/docs` - Interactive API documentation

### 3. Testing the Server

```bash
python test_mlx_server.py
```

## Training Configuration

Key parameters in `config.yaml`:
- **Model**: Phi-3-mini-4k-instruct (4-bit quantized)
- **LoRA Rank**: 16 (memory efficient)
- **Batch Size**: 2 (with gradient accumulation of 4)
- **Learning Rate**: 5e-5
- **Max Steps**: 1000

## Memory Usage

On M1 Max with 32GB RAM:
- Training: ~12-16GB
- Inference: ~6-8GB
- Speed: 15-20 tokens/second

## Integration with Next.js

The MLX server can be integrated as an alternative to the hybrid autocomplete:

1. Start the MLX server on port 8003
2. Update `ai-text.ts` to check MLX endpoint when `USE_MLX_MODEL=true`
3. Compare performance with existing hybrid system

## Model Quality

The fine-tuned model specializes in:
- Bio-style text completions
- Natural sentence flow
- Domain-specific vocabulary
- Appropriate tone and style

## Performance Metrics

### Training Results (M1 Max 32GB)
- **Training Time**: 3.4 minutes for 1000 steps
- **Final Validation Loss**: 2.457
- **Test Loss**: 2.411
- **Test Perplexity**: 11.146
- **Training Speed**: ~6 iterations/second
- **Peak Memory**: 3.2GB

### Inference Performance
- **Single Request**: 340ms average (300-400ms range)
- **Median Response**: 339.9ms
- **95th Percentile**: 341.3ms
- **Throughput**: 3.0 requests/second
- **Batch Processing**: ~530ms per prompt in batch mode

### Model Improvements Observed
- Context-aware completions matching bio style
- Clean output without special tokens
- Consistent response times
- Good handling of various prompt types

## Troubleshooting

1. **Out of Memory**: Reduce batch size in config.yaml
2. **Slow Training**: Normal on first run (downloading model)
3. **Import Errors**: Ensure MLX is installed: `pip install mlx mlx-lm`
4. **Temperature Issues**: Use make_sampler from mlx_lm.sample_utils

## Next Steps

1. ✅ Train the model with your bio data
2. ✅ Test the server with various prompts
3. ✅ Benchmark performance metrics
4. Compare quality with hybrid autocomplete system on port 8001
5. Choose the best performing system for production use