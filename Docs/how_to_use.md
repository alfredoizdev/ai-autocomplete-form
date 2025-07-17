# How to Use AI-Train-LLM Application

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Getting Started](#getting-started)
4. [Vector Database Setup](#vector-database-setup)
5. [Training Custom Models](#training-custom-models)
6. [Running the Application](#running-the-application)
7. [API Endpoints](#api-endpoints)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

## Overview

This application provides AI-powered text autocomplete functionality optimized for personal bio completion. It uses a sophisticated hybrid approach combining:

- **Vector Database (ChromaDB)**: Fast similarity search for relevant context
- **Large Language Model (Ollama)**: Gemma 3 12B for intelligent completions
- **Custom Fine-tuned Models**: Specialized models trained on your specific data
- **Real-time Suggestions**: Smart debouncing and spell-check integration

## System Architecture

### Components

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Next.js App   │────▶│  Python API     │────▶│   ChromaDB      │
│   (Port 3000)   │     │  (Port 8001)    │     │ (Vector Store)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │     Ollama      │
                        │  (Port 11434)   │
                        └─────────────────┘
```

### Data Flow

1. User types in the web interface
2. After 1.5-second pause, autocomplete triggers
3. Python API receives the prompt
4. Vector search finds similar contexts in ChromaDB
5. Ollama generates completions using the context
6. Top suggestions are returned to the user

## Getting Started

### Prerequisites

- **Node.js 18+** and npm
- **Python 3.8+**
- **Docker** (optional, for containerized services)
- **Ollama** installed locally
- **32GB RAM** recommended for training

### Initial Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-train-llm
   ```

2. **Install frontend dependencies**:
   ```bash
   npm install
   ```

3. **Set up Python environment**:
   ```bash
   cd python
   python3 -m venv venv
   source venv/bin/activate  # On Mac/Linux
   pip install -r requirements.txt
   ```

4. **Install and configure Ollama**:
   ```bash
   # Install Ollama (Mac)
   brew install ollama
   
   # Start Ollama service
   ollama serve
   
   # Pull the Gemma model
   ollama pull gemma3:12b
   ```

5. **Set up environment variables**:
   ```bash
   # In project root, create .env.local
   echo "OLLAMA_PATH_API=http://127.0.0.1:11434/api" > .env.local
   ```

## Vector Database Setup

### Initial ChromaDB Setup

1. **Navigate to the vector database directory**:
   ```bash
   cd python/vector_db
   ```

2. **Initialize the database with existing data**:
   ```bash
   python setup_chromadb.py
   ```
   
   This will:
   - Load bio data from `data/bio.json`
   - Create embeddings for each text
   - Store in ChromaDB with metadata
   - Create searchable index

3. **Verify the setup**:
   ```bash
   python test_chromadb.py
   ```

### Using Custom Data for Vector Database

1. **Prepare your data** in JSON format:
   ```json
   [
     "Your first bio text here...",
     "Another bio example...",
     "More training data..."
   ]
   ```

2. **Update the data file**:
   ```bash
   cp your_data.json data/bio.json
   ```

3. **Re-run setup**:
   ```bash
   python setup_chromadb.py --reset  # Clears existing data
   ```

## Training Custom Models

### Method 1: Quick Training (Recommended for Beginners)

1. **Prepare your training data**:
   ```bash
   cd python/mlx_training
   
   # Edit prepare_mlx_data.py to point to your data file
   # Then run:
   python prepare_mlx_data.py
   ```

2. **Run simple training**:
   ```bash
   python train_simple.py
   ```
   
   This uses:
   - Small subset for testing (100 examples)
   - Conservative memory settings
   - ~15-30 minutes training time

3. **Test the model**:
   ```bash
   python test_model.py --model simple_output
   ```

### Method 2: Full Training (Better Quality)

1. **Prepare data** (same as above)

2. **Run full training**:
   ```bash
   python train_bio_improved.py
   ```
   
   Configuration options:
   ```python
   # In train_bio_improved.py
   config = TrainingConfig(
       batch_size=2,              # Lower if out of memory
       learning_rate=5e-5,        # Adjust for convergence
       num_train_epochs=2,        # More = better but longer
       max_seq_length=128,        # Maximum text length
       gradient_accumulation_steps=8,  # Effective batch size
   )
   ```

3. **Monitor training**:
   - Watch loss values (should decrease)
   - Check sample generations
   - Typical time: 2-4 hours

### Method 3: MLX Training (Apple Silicon Optimized)

1. **Download base model**:
   ```bash
   python download_model.py
   ```

2. **Train with MLX**:
   ```bash
   python train_mlx.py
   ```

3. **Convert for Ollama** (optional):
   ```bash
   python convert_to_gguf.py
   ollama create bio-model -f Modelfile
   ```

### Training Tips

- **Data Quality**: Clean, consistent data gives better results
- **Data Quantity**: Minimum 100-500 examples, ideally 1000+
- **Memory Issues**: Reduce batch_size and max_seq_length
- **Poor Results**: Train longer or use more data

## Running the Application

### Start All Services

1. **Start Ollama** (in terminal 1):
   ```bash
   ollama serve
   ```

2. **Start Python API** (in terminal 2):
   ```bash
   cd python/api
   source ../venv/bin/activate
   python api_server.py
   # Or with auto-reload:
   uvicorn api_server:app --reload --port 8001
   ```

3. **Start Next.js** (in terminal 3):
   ```bash
   npm run dev
   ```

4. **Access the application**:
   - Frontend: http://localhost:3000
   - API Docs: http://localhost:8001/docs

### Docker Setup (Optional)

```bash
# Start supporting services
docker-compose up -d

# This starts:
# - Weaviate (alternative vector DB)
# - Transformers inference API
```

## API Endpoints

### Main API (Port 8001)

- **GET /** - Health check
- **POST /api/autocomplete** - Vector-only suggestions
  ```json
  {
    "prompt": "I am a software",
    "max_suggestions": 3
  }
  ```

- **POST /api/autocomplete/hybrid** - Vector + LLM suggestions
  ```json
  {
    "prompt": "We are looking for",
    "max_suggestions": 3,
    "temperature": 0.8
  }
  ```

- **GET /api/stats** - Database statistics

### Response Format

```json
{
  "suggestions": [
    {
      "text": "engineer with a passion for innovation",
      "score": 0.95,
      "source": "vector_exact"
    },
    {
      "text": "fun and exciting adventures together",
      "score": 0.87,
      "source": "llm_generated"
    }
  ],
  "metadata": {
    "processing_time": 0.124,
    "method": "hybrid"
  }
}
```

## Troubleshooting

### Common Issues

1. **"Ollama not running"**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # If not, start it:
   ollama serve
   ```

2. **"Out of memory" during training**
   ```python
   # Reduce these in training config:
   batch_size = 1
   max_seq_length = 64
   gradient_accumulation_steps = 4
   ```

3. **"ChromaDB connection error"**
   ```bash
   # Reset the database
   cd python/vector_db
   rm -rf chroma_data/
   python setup_chromadb.py
   ```

4. **"Poor autocomplete quality"**
   - Train with more data
   - Increase training epochs
   - Adjust temperature in API calls

### Performance Optimization

1. **Slow autocomplete response**:
   - Reduce number of vector search results
   - Use smaller context window
   - Enable caching in API

2. **High memory usage**:
   - Limit ChromaDB collection size
   - Use quantized models
   - Implement request queuing

## Best Practices

### Data Preparation

1. **Quality over Quantity**:
   - Clean, well-formatted text
   - Remove duplicates
   - Consistent style and tone

2. **Data Structure**:
   - Minimum 50 words per entry
   - Complete thoughts/sentences
   - Diverse examples

### Training

1. **Start Small**:
   - Test with 100 examples first
   - Validate the pipeline works
   - Then scale up

2. **Monitor Progress**:
   - Save checkpoints regularly
   - Test on validation set
   - Check sample outputs

3. **Hyperparameter Tuning**:
   - Start with defaults
   - Adjust one parameter at a time
   - Keep notes on what works

### Production Deployment

1. **API Configuration**:
   ```python
   # In api_server.py
   MAX_CONCURRENT_REQUESTS = 10
   REQUEST_TIMEOUT = 30
   CACHE_TTL = 3600
   ```

2. **Monitoring**:
   - Set up logging
   - Track API response times
   - Monitor memory usage

3. **Scaling**:
   - Use Redis for caching
   - Deploy API behind load balancer
   - Consider GPU for inference

### Security

1. **Input Validation**:
   - Sanitize user inputs
   - Limit prompt length
   - Rate limit API calls

2. **Data Privacy**:
   - Don't log sensitive prompts
   - Implement user authentication
   - Use HTTPS in production

## Advanced Configuration

### Custom Ollama Models

1. **Create Modelfile**:
   ```dockerfile
   FROM gemma3:12b
   PARAMETER temperature 0.8
   PARAMETER top_p 0.9
   PARAMETER repeat_penalty 1.1
   SYSTEM "You are an AI assistant specialized in completing personal bios."
   ```

2. **Build and run**:
   ```bash
   ollama create bio-assistant -f Modelfile
   ollama run bio-assistant
   ```

### Fine-tuning Parameters

```python
# Advanced training configuration
config = TrainingConfig(
    # Model settings
    model_name="gpt2-medium",  # or "distilgpt2", "gpt2-large"
    
    # Training hyperparameters
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    
    # LoRA settings
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.1,
    
    # Data settings
    max_train_samples=10000,
    validation_split=0.1,
    
    # Hardware optimization
    fp16=True,  # Mixed precision training
    gradient_checkpointing=True,  # Save memory
)
```

## Contributing

When adding new features:

1. Update training data format if needed
2. Test with small dataset first
3. Document any new dependencies
4. Update this guide with new instructions

---

For more help, check the [API_SERVER_GUIDE.md](API_SERVER_GUIDE.md) or open an issue on GitHub.