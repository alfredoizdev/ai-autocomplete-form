# MLX Bio Autocomplete Server

This directory contains the MLX-based model serving infrastructure for the bio autocomplete feature.

## Overview

The MLX server provides fast inference for fine-tuned Llama 3.2 models on Apple Silicon. It supports multiple models with automatic priority-based selection, optimized for bio text completion.

## Current Model Hierarchy

The server automatically selects the best available model in this order:

1. **HIGH-QUALITY Llama-3.2-3B-Instruct** (Primary)
   - Grammar-filtered LookingFor dataset
   - 2000 iterations, learning rate 1e-5
   - Location: `models/lookingfor-llama3-3b-hq-lora/`
   - Best quality results

2. **Standard Llama-3.2-3B-Instruct** 
   - LookingFor dataset, 1500 iterations
   - Location: `models/lookingfor-llama3-3b-lora/`

3. **Llama-3.2-1B-Instruct** (Faster alternative)
   - LookingFor dataset, smaller model
   - Location: `models/lookingfor-llama3-lora/`

4. **Bio-sentence models** (Legacy fallbacks)
   - Older training approach
   - Kept for backward compatibility

5. **Phi-3 model** (Legacy)
   - Original model, causes repetition issues
   - Location: `mlx_server/models/bio-phi3-lora/`

## Directory Structure

```
mlx_server/
├── mlx_model_server.py    # FastAPI server (port 8003)
├── benchmark_mlx.py       # Performance benchmarking
├── debug_completion.py    # Debug tool for testing
├── mlx_server.log        # Server logs
├── config.yaml           # Legacy training config
├── data/                 # Legacy training data
└── requirements.txt      # Python dependencies
```

## Quick Start

### Starting the Server

```bash
# Use the provided startup script (recommended)
./start_trained.sh

# Or start manually
cd python/mlx_server
python mlx_model_server.py
```

The server runs on port 8003 and provides:
- `/api/autocomplete/mlx` - Single completion endpoint
- `/api/autocomplete/mlx/batch` - Batch completions
- `/health` - Health check
- `/docs` - Interactive API documentation

### Testing the Server

```bash
# Debug a specific completion
python debug_completion.py

# Run performance benchmarks
python benchmark_mlx.py
```

## API Usage

### Single Completion
```python
POST http://localhost:8003/api/autocomplete/mlx
{
    "prompt": "Looking for fun loving people",
    "max_tokens": 50,
    "temperature": 0.7,
    "stop": [".", "!", "?", "\n"]
}
```

### Response
```json
{
    "completion": "that we can have fun with in and out of the bedroom",
    "elapsed_ms": 125.4,
    "model_name": "Llama-3.2-3B-hq (LoRA)"
}
```

## Performance Metrics

### HIGH-QUALITY Llama 3B Model (M1/M2/M3)
- **Response Time**: 100-150ms average
- **Memory Usage**: ~4-6GB
- **Throughput**: 8-10 requests/second
- **Quality**: Superior grammar and coherence

### Standard Llama 3B Model
- **Response Time**: 100-150ms average
- **Memory Usage**: ~4-6GB
- **Quality**: Good, occasional grammar issues

### Llama 1B Model (Faster Option)
- **Response Time**: 50-100ms average
- **Memory Usage**: ~2-3GB
- **Quality**: Decent, more concise responses

## Integration with Next.js

The MLX server integrates with the main app through:

1. **Environment Variable**: `AUTOCOMPLETE_MODE=trained`
2. **Action Handler**: `actions/ai-text.ts` checks mode and routes to MLX
3. **Startup Script**: `./start_trained.sh` sets everything up

## Model Features

The fine-tuned models specialize in:
- Bio-style text completions with lifestyle vocabulary
- Natural sentence flow and completion
- Grammar-aware predictions (HIGH-QUALITY model)
- Fast response times on Apple Silicon
- Consistent tone and style

## Advanced Configuration

### Custom Model Loading
To use a specific model, modify the model loading priority in `mlx_model_server.py`.

### Temperature Tuning
- Lower (0.3-0.5): More predictable, common phrases
- Medium (0.6-0.8): Balanced creativity (default: 0.7)
- Higher (0.9-1.0): More creative, varied responses

### Stop Tokens
Default stop tokens: `[".", "!", "?", "\n"]`
Customize based on your completion needs.

## Troubleshooting

1. **Model Not Found**: Ensure model files exist in the expected locations
2. **Out of Memory**: Use the 1B model for lower memory usage
3. **Slow First Request**: Model loading takes 5-10 seconds initially
4. **Import Errors**: Run `pip install -r requirements.txt`

## Training New Models

To train your own model, use the scripts in `python/mlx_training/`:
- `prepare_hq_data_fast.sh` - Prepare high-quality training data
- `start_training_mlx_community.sh` - Train with optimized parameters

## Recent Updates

- Removed legacy training scripts (`train.py`, `convert_to_mlx_format.py`)
- Updated to use Llama 3.2 models exclusively
- Added HIGH-QUALITY grammar-filtered model as primary
- Improved error handling and logging
- Better integration with startup scripts