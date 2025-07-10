# Training Custom LLM for Bio Autocomplete - Complete Beginner's Guide

This guide provides a step-by-step approach to training your own custom Language Learning Model (LLM) specifically for realtime autocomplete in bio profiles, and deploying it locally using Ollama.

## Overview

**Project Goal**: Create a specialized LLM that provides intelligent, context-aware autocomplete suggestions for bio profiles with authentic language patterns and terminology.

**Important Note**: Ollama does NOT support fine-tuning models directly. This guide uses Unsloth for training (beginner-friendly, 60% less VRAM) and then imports the trained model to Ollama for local inference.

## What You'll Build

1. **Custom Bio Model**: Fine-tuned Gemma 3:12B with your bio.json data using Unsloth
2. **Vector Database**: Local, secure storage of profile patterns for better context
3. **Enhanced API**: Combines vector context + your custom model for superior suggestions
4. **Complete Integration**: Works with your existing Next.js autocomplete app

**Everything stays local and secure on your machine - no data ever leaves your computer!**

## Training Workflow Overview

1. **Prepare Data**: Format your bio.json for training
2. **Fine-tune with Unsloth**: Train Gemma3:12b on your data (uses 60% less VRAM)
3. **Export to GGUF**: Convert trained model for Ollama
4. **Import to Ollama**: Create custom model in Ollama
5. **Integrate**: Use with your Next.js app

## Phase 1: Environment Setup & Planning

### Task 1.1: Choose Your Training Tool

- [ ] **Unsloth** (Recommended): Beginner-friendly, uses 60% less VRAM, works with free Google Colab
- [ ] **Hugging Face Transformers** (Alternative): More control but requires more technical knowledge
- [ ] **AutoTrain** (Simple but limited): Good for very basic use cases

### Task 1.2: Select Base Model for Autocomplete

- [ ] **Your Choice**: Gemma 3:12B (Perfect for fine-tuning and autocomplete)
- [ ] **Alternative**: Llama 3.1 8B (If you need faster inference) 
- [ ] **Budget Option**: Gemma 2 2B (Works on smaller GPUs)
- [ ] **Avoid**: Large 70B+ models (Too slow for realtime autocomplete)

**Note**: We'll use Unsloth to fine-tune Gemma 3:12B, then import to Ollama for inference.

### Task 1.3: Hardware Requirements Check

- [ ] **With Unsloth**: Only 15GB VRAM needed for Gemma 3:12B (vs 24GB+ normally)
- [ ] **Free Option**: Google Colab Free tier (T4 GPU with 15GB VRAM) works!
- [ ] **Local GPU**: RTX 3060 12GB or better
- [ ] **System RAM**: 16GB minimum, 32GB recommended
- [ ] **Storage**: ~50GB free space for models and data

### Task 1.4: Install Required Tools

**Step 1: Install Ollama for Inference (After Training)**

```bash
# Install Ollama for running the trained model
# Download from https://ollama.ai or use:
curl -fsSL https://ollama.ai/install.sh | sh

# Test Ollama is working
ollama serve  # Start Ollama server
```

**Step 2: Install Training Tools**

```bash
# Option A: Local Installation
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install chromadb sentence-transformers fastapi uvicorn

# Option B: Google Colab (Recommended for beginners)
# Just open a new Colab notebook and run:
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes
```

## Phase 2: Bio Autocomplete Data Preparation

### Task 2.1: Prepare Your Bio Data

Create `prepare_bio_data.py`:

```python
import json
from datasets import Dataset

def prepare_autocomplete_data(bio_file_path):
    """
    Convert bio.json to training format for autocomplete
    """
    # Load your bio data
    with open(bio_file_path, 'r') as f:
        bios = json.load(f)
    
    training_examples = []
    
    for bio in bios:
        # Assuming bio has 'text' field - adjust based on your structure
        text = bio.get('text', bio.get('bio', str(bio)))
        
        # Create multiple training examples from each bio
        words = text.split()
        
        # Generate examples at different completion points
        for i in range(3, len(words)-2, 2):  # Start from 3 words
            prompt = " ".join(words[:i])
            completion = " ".join(words[i:min(i+5, len(words))])  # Next 5 words
            
            # Simple format for text completion
            training_examples.append({
                "text": f"{prompt} {completion}"
            })
    
    # Convert to dataset
    dataset = Dataset.from_list(training_examples)
    print(f"Created {len(training_examples)} training examples")
    
    return dataset

# Example usage
train_dataset = prepare_autocomplete_data('data/bio.json')
```

### Task 2.2: Data Quality Tips

- [ ] **Minimum Data**: At least 100 bios, ideally 500+
- [ ] **Clean Text**: Remove special characters, excessive punctuation
- [ ] **Varied Examples**: Include diverse bio styles and lengths
- [ ] **Test Split**: Keep 10% of data for validation

## Phase 3: Fine-Tuning with Unsloth (Beginner-Friendly)

### Task 3.1: Complete Training Script

Create `train_bio_model.py` or use this in Google Colab:

```python
from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments
from trl import SFTTrainer

# 1. Load Model with Unsloth (uses only 15GB VRAM!)
max_seq_length = 512  # Shorter for autocomplete
dtype = None  # Auto-detect 
load_in_4bit = True  # CRITICAL: Reduces memory from 24GB to 15GB

print("Loading Gemma3:12b with Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-3-12b",  # Base model, not instruction-tuned
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Apply LoRA for efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # Good for autocomplete tasks
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,  # Unsloth optimized
    bias = "none",
    use_gradient_checkpointing = "unsloth",  # Saves memory
    random_state = 3407,
)

print("Model loaded! Using only ~15GB VRAM")

# 2. Prepare your data
from prepare_bio_data import prepare_autocomplete_data
train_dataset = prepare_autocomplete_data('data/bio.json')

# 3. Set training parameters
training_args = TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 10,
    num_train_epochs = 1,  # Start with 1 epoch
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 10,
    optim = "adamw_8bit",
    seed = 3407,
    output_dir = "outputs",
    save_strategy = "epoch",
)

# 4. Create trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = training_args,
)

# 5. Start training (10-30 minutes on T4 GPU)
print("Starting training...")
trainer_stats = trainer.train()
print(f"Training completed! Loss: {trainer_stats.training_loss}")

# 6. Save the fine-tuned model
model.save_pretrained("bio-completion-gemma")
tokenizer.save_pretrained("bio-completion-gemma")

# 7. Export to GGUF for Ollama
print("Exporting to GGUF format...")
model.save_pretrained_gguf(
    "bio-completion-gguf",
    tokenizer,
    quantization_method = ["q4_k_m", "q8_0"]  # Two sizes
)

print("✅ Model exported! Files saved in bio-completion-gguf/")
```

### Task 3.2: Quick Test Your Model

Add this to your training script:

```python
# Test the model before export
def test_completion(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=20, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test examples
print("\n=== Testing completions ===")
print(test_completion("I am a software engineer with"))
print(test_completion("Looking for new friends who"))
print(test_completion("My hobbies include"))
```

## Phase 4: Import to Ollama for Local Deployment

### Task 4.1: Create Ollama Modelfile

After training, create `Modelfile`:

```dockerfile
# Point to your exported GGUF file
FROM ./bio-completion-gguf/bio-completion-gguf-q4_k_m.gguf

# Template for completion (not chat)
TEMPLATE """{{ .Prompt }}"""

# Parameters optimized for autocomplete
PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER stop " "
PARAMETER stop "."
PARAMETER stop "\n"
```

### Task 4.2: Import to Ollama

```bash
# Create your custom model in Ollama
ollama create bio-autocomplete -f Modelfile

# Test it
ollama run bio-autocomplete "I am a software developer who"
```

### Task 4.3: Verify Model Quality

```bash
# Test various prompts
ollama run bio-autocomplete "My interests include"
ollama run bio-autocomplete "I work as a"
ollama run bio-autocomplete "Looking for people who"

# Check response time
time ollama run bio-autocomplete "I enjoy"
```

## Phase 5: Integration with Next.js App

### Task 5.1: Update Your API Action

Update `actions/ai-text.ts`:

```typescript
export async function askOllamaCompletationAction(prompt: string) {
  try {
    const response = await fetch(`${process.env.OLLAMA_PATH_API}/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "bio-autocomplete",  // Your custom model
        prompt: prompt,
        stream: false,
        options: {
          temperature: 0.7,
          top_p: 0.9,
          num_predict: 20,  // Tokens to generate
          stop: [" ", ".", "\n"],  // Stop at these tokens
        },
      }),
    });

    const data = await response.json();
    return data.response.trim();
  } catch (error) {
    console.error("Ollama API error:", error);
    return "";
  }
}
```

### Task 5.2: Start Services

```bash
# Terminal 1: Ollama server
ollama serve

# Terminal 2: Your Next.js app
npm run dev
```

## Quick Reference Guide

### Complete Workflow Commands

```bash
# 1. Prepare environment (Google Colab recommended)
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 2. Train model (run train_bio_model.py)
python train_bio_model.py

# 3. Create Ollama model
ollama create bio-autocomplete -f Modelfile

# 4. Test
ollama run bio-autocomplete "I am interested in"

# 5. Use in your app
# Update actions/ai-text.ts to use "bio-autocomplete" model
```

### Memory Requirements Summary

| Method | VRAM Required | Speed |
|--------|--------------|-------|
| Standard Fine-tuning | 24GB+ | Slow |
| Unsloth Fine-tuning | 15GB | 1.6x faster |
| Unsloth + Colab Free | 15GB (T4) | Good |
| Inference Only | 8GB | Fast |

### Common Issues & Solutions

1. **Out of Memory**: 
   - Use smaller batch size (1 instead of 2)
   - Enable gradient checkpointing
   - Use Google Colab if local GPU insufficient

2. **Poor Completions**:
   - Train for 2-3 epochs instead of 1
   - Add more bio examples
   - Adjust temperature (0.5-0.9)

3. **Slow Training**:
   - Normal: 10-30 mins for 1 epoch
   - Use packing=True for short sequences
   - Reduce max_seq_length to 256

4. **Import to Ollama Fails**:
   - Check GGUF file path is correct
   - Ensure Ollama server is running
   - Try the q8_0 quantization instead

## Success Metrics

After completing this guide:
- ✅ Custom model trained on your bio data
- ✅ 60% less VRAM usage with Unsloth
- ✅ <500ms autocomplete response time
- ✅ Better, more relevant suggestions
- ✅ Everything running locally and securely

## Next Steps

1. **Collect More Data**: Add more bios for better quality
2. **A/B Testing**: Compare with base model
3. **Fine-tune Parameters**: Adjust temperature, top_p
4. **Add Vector Database**: Enhanced context (see original guide)

Remember: Start simple, test often, and iterate based on results!