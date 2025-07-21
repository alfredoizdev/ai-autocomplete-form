# MLX Training Plan for MacBook Pro M1 Max 32GB RAM

## Executive Summary

This plan outlines a step-by-step approach to successfully train a local LLM using MLX on your MacBook Pro M1 Max with 32GB RAM. Based on current MLX capabilities and best practices from 2025, we'll use LoRA fine-tuning for memory efficiency and leverage the existing bio dataset infrastructure.

## Hardware Capability Assessment

Your M1 Max with 32GB RAM can comfortably handle:
- **7B-13B parameter models** with LoRA fine-tuning
- **Smaller models (2B-7B)** with full fine-tuning
- **Quantized versions** of larger models (up to 30B with 4-bit quantization)

## Step-by-Step Training Plan

### Phase 1: Environment Setup and Dependencies

1. **Install/Update MLX and MLX-LM**
   ```bash
   pip install --upgrade mlx mlx-lm
   pip install transformers datasets accelerate
   ```

2. **Verify MLX Installation**
   ```bash
   python -c "import mlx; print(mlx.__version__)"
   python -c "import mlx_lm; print('MLX-LM ready')"
   ```

3. **Setup Hugging Face CLI** (for model downloads)
   ```bash
   pip install huggingface-hub
   huggingface-cli login
   ```

### Phase 2: Model Selection and Download

Based on your 32GB RAM, recommended models:

1. **Primary Option: Mistral 7B** (Best balance)
   - Size: ~14GB (7B params)
   - Performance: Excellent
   - Training time: 3-4 hours

2. **Alternative Options:**
   - **Gemma 2B**: Faster training (1-2 hours), lower quality
   - **Phi-3 Mini**: 3.8B params, good for specific tasks
   - **Llama 3 8B**: Latest architecture, higher quality

3. **Download Command:**
   ```bash
   python -m mlx_lm.convert --hf-path mistralai/Mistral-7B-Instruct-v0.2 -q
   ```

### Phase 3: Data Preparation

1. **Leverage Existing Bio Dataset**
   - Already have 4,599 training examples in `bio_dataset/`
   - Format: prompt-completion pairs
   - Split: 90% train, 10% validation

2. **Convert to MLX Format** (if needed)
   ```python
   # Create train.jsonl, valid.jsonl, test.jsonl
   # Format: {"text": "Complete: {prompt} → {completion}"}
   ```

### Phase 4: Training Configuration

1. **Create Optimized Config File** (`lora_config.yaml`):
   ```yaml
   model: "mistralai/Mistral-7B-Instruct-v0.2"
   train: true
   data: "./bio_dataset"
   
   # LoRA parameters optimized for 32GB RAM
   lora_layers: 16
   lora_parameters:
     rank: 8
     alpha: 16
     dropout: 0.05
   
   # Training parameters
   batch_size: 2
   gradient_accumulation_steps: 4
   learning_rate: 2e-4
   num_epochs: 1
   
   # Memory optimization
   max_seq_length: 512
   gradient_checkpointing: true
   
   # Checkpointing
   save_every: 250
   eval_steps: 100
   
   # Output
   adapter_path: "./adapters/bio_mistral_lora"
   ```

### Phase 5: Training Execution

1. **Run Training Script**
   ```bash
   python -m mlx_lm.lora \
     --config lora_config.yaml \
     --train \
     --wandb bio-autocomplete-training
   ```

2. **Monitor Training**
   - Watch memory usage in Activity Monitor
   - Expect 20-25GB peak memory usage
   - Training speed: ~50-100 tokens/sec

3. **Backup Checkpoints**
   - Checkpoints saved every 250 steps
   - Can resume with `--resume-adapter-file`

### Phase 6: Model Evaluation and Testing

1. **Run Evaluation**
   ```bash
   python -m mlx_lm.lora \
     --model mistralai/Mistral-7B-Instruct-v0.2 \
     --adapter-path ./adapters/bio_mistral_lora \
     --data ./bio_dataset \
     --test
   ```

2. **Interactive Testing**
   ```python
   from mlx_lm import load, generate
   
   model, tokenizer = load(
       "mistralai/Mistral-7B-Instruct-v0.2",
       adapter_path="./adapters/bio_mistral_lora"
   )
   
   prompt = "We are a couple who"
   response = generate(model, tokenizer, prompt=prompt, max_tokens=50)
   print(response)
   ```

### Phase 7: Integration with Existing System

1. **Update API Server** (`api_server.py`):
   - Add MLX model as additional backend
   - Implement fallback logic
   - Cache MLX responses

2. **Performance Optimization**:
   - Use 4-bit quantization for faster inference
   - Implement batching for multiple requests
   - Add model warmup on server start

3. **A/B Testing Setup**:
   - Route 10% traffic to MLX model initially
   - Compare response quality and latency
   - Gradually increase if performance is good

## Memory Optimization Techniques

1. **Gradient Checkpointing**
   ```python
   from mlx.nn import checkpoint
   # Apply to transformer blocks
   ```

2. **Mixed Precision Training**
   - MLX handles this automatically for Apple Silicon

3. **Dynamic Batch Sizing**
   - Start with batch_size=1 if OOM
   - Use gradient accumulation for effective larger batches

4. **Quantization Options**
   - 4-bit quantization reduces memory by 75%
   - QLoRA allows training quantized models

## Troubleshooting Guide

### Out of Memory Errors
1. Reduce `lora_layers` to 8 or 4
2. Decrease `max_seq_length` to 256
3. Use batch_size=1 with more gradient accumulation
4. Try 4-bit quantized base model

### Slow Training
1. Ensure MLX is using GPU (check Activity Monitor)
2. Reduce logging frequency
3. Use smaller model (Gemma 2B)
4. Disable gradient checkpointing if memory allows

### Poor Model Quality
1. Increase training epochs to 2-3
2. Try higher LoRA rank (16 or 32)
3. Adjust learning rate (try 1e-4 or 5e-5)
4. Add more training data

## Expected Outcomes

- **Training Time**: 3-4 hours for Mistral 7B
- **Memory Usage**: 20-25GB peak
- **Model Size**: ~200MB adapter files
- **Inference Speed**: 50-100 tokens/second
- **Quality**: Significant improvement for bio completions

## Next Steps After Training

1. Convert to GGUF format for Ollama (optional)
2. Deploy to production API server
3. Monitor performance metrics
4. Collect user feedback for iteration
5. Consider training larger model if results are good

## Alternative Approaches

1. **Distributed Training** (if need larger models):
   - Use `mx.distributed` for multi-GPU
   - Split model across devices

2. **Progressive Training**:
   - Start with Gemma 2B for quick iteration
   - Move to larger models based on results

3. **Ensemble Approach**:
   - Train multiple smaller models
   - Combine outputs for better quality

This plan is designed to be executed incrementally with checkpoints at each phase to ensure success on your M1 Max hardware.