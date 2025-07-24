# Configuration Guide

Complete guide to all configuration options in the AI Bio Autocomplete system.

## 🔧 Environment Variables

### Core Settings (.env.local)

```bash
# Autocomplete mode selection
AUTOCOMPLETE_MODE=hybrid          # Options: 'hybrid' or 'trained'

# Ollama API endpoint
OLLAMA_PATH_API=http://127.0.0.1:11434/api

# Optional: OpenAI fallback
NEXT_PUBLIC_OPENAI_API_KEY=your-key-here

# Optional: Custom ports
API_SERVER_PORT=8001              # Python API server
MLX_SERVER_PORT=8003              # MLX model server
```

### Development Settings

```bash
# Enable/disable features
NEXT_PUBLIC_ENABLE_SPELL_CHECK=true
NEXT_PUBLIC_ENABLE_KICK_DETECTION=true
NEXT_PUBLIC_ENABLE_CACHING=true

# Debug mode
NEXT_PUBLIC_DEBUG=false           # Enable debug logging
```

## 🎛 Application Configuration

### Autocomplete Settings

Located in `hooks/useFormAutocomplete.tsx`:

```typescript
// Debounce timing
const DEBOUNCE_MS = 1500;         // Wait before triggering autocomplete

// Trigger thresholds
const MIN_WORDS_FOR_AUTOCOMPLETE = 3;
const MIN_CHARS_FOR_AUTOCOMPLETE = 10;

// Response settings
const MAX_SUGGESTIONS = 3;
const TARGET_WORD_COUNT = "8-20"; // Words per suggestion
```

### Spell Check Configuration

Located in `hooks/useSpellCheck.tsx`:

```typescript
// Performance settings
const SPELL_CHECK_DEBOUNCE = 800; // ms
const MIN_WORD_LENGTH = 3;        // Skip short words

// Dictionary settings
const CUSTOM_DICT_KEY = 'customDictionary';
const MAX_CUSTOM_WORDS = 1000;
```

### Kick Detection Patterns

Located in `lib/kickDetection.ts`:

```typescript
// Pattern categories (40+ patterns)
const KICK_PATTERNS = {
  direct: [...],          // Direct mentions
  phonetic: [...],        // Sound-alike variations
  leet: [...],           // L33t speak
  spaced: [...],         // S p a c e d
  reversed: [...],       // kcik
  unicode: [...]         // Special characters
};

// Performance threshold
const MAX_CHECK_LENGTH = 10000;   // Characters
```

## 🖥 Server Configuration

### Python API Server (api_server.py)

```python
# Server settings
HOST = "0.0.0.0"
PORT = 8001
RELOAD = False                    # Hot reload disabled

# Ollama settings
OLLAMA_API_URL = os.getenv("OLLAMA_PATH_API", "http://127.0.0.1:11434/api")
OLLAMA_MODEL = "gemma3:12b"
OLLAMA_TIMEOUT = 10.0            # seconds

# Generation parameters
TEMPERATURE = 0.85
TOP_P = 0.95
MAX_TOKENS = 25                  # Limit response length

# Cache settings
CACHE_TTL = 300                  # 5 minutes
```

### MLX Server (mlx_model_server.py)

```python
# Model priority order
MODEL_PRIORITY = [
    "lookingfor-llama3-3b-hq-lora",     # HIGH-QUALITY
    "lookingfor-llama3-3b-lora",        # Standard
    "lookingfor-llama3-lora",           # 1B faster
    "bio-llama3-lora",                  # Bio dataset
    "phi3-lora"                         # Legacy
]

# Generation settings
MAX_TOKENS = 50
TEMPERATURE = 0.7
TOP_P = 0.95
REPETITION_PENALTY = 1.1
```

### ChromaDB Configuration

Located in `vector_search.py`:

```python
# Collection settings
COLLECTION_NAME = "bio_embeddings"
EMBEDDING_FUNCTION = "default"    # Sentence transformers

# Search parameters
N_RESULTS = 5                    # Similar documents
MIN_RELEVANCE_SCORE = 0.7       # Similarity threshold
```

## 🚀 Performance Tuning

### Frontend Optimization

```typescript
// Adaptive debouncing (useFormAutocomplete)
const getAdaptiveDebounce = (textLength: number) => {
  if (textLength < 50) return 50;
  if (textLength < 100) return 100;
  if (textLength < 200) return 200;
  return 400;
};
```

### Backend Optimization

```python
# Connection pooling
HTTPX_LIMITS = httpx.Limits(
    max_connections=100,
    max_keepalive_connections=20
)

# ChromaDB settings
CHROMA_CACHE_SIZE = 1000        # Cached embeddings
BATCH_SIZE = 100                # Batch processing
```

## 🎨 UI Configuration

### Tailwind CSS (v4)

Located in `tailwind.config.ts`:

```javascript
{
  theme: {
    extend: {
      colors: {
        primary: {...},
        secondary: {...}
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'pulse-subtle': 'pulse 3s infinite'
      }
    }
  }
}
```

### Component Settings

```typescript
// Form.tsx
const MAX_BIO_LENGTH = 500;
const PLACEHOLDER_TEXT = "I am";
const MOBILE_FONT_SIZE = "16px"; // Prevent zoom

// SpellCheckPopup.tsx
const POPUP_OFFSET = 8;          // pixels
const MAX_SUGGESTIONS = 5;
```

## 📊 Monitoring Configuration

### Logging

```python
# API Server logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/api_server.log'),
        logging.StreamHandler()
    ]
)
```

### Metrics

```python
# Track performance
METRICS = {
    'autocomplete_requests': 0,
    'average_response_time': 0,
    'cache_hits': 0,
    'cache_misses': 0
}
```

## 🔒 Security Configuration

### CORS Settings

```python
# api_server.py
CORS_ORIGINS = ["http://localhost:3000"]
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]
```

### Input Validation

```python
# Request limits
MAX_PROMPT_LENGTH = 500
MIN_PROMPT_LENGTH = 3
RATE_LIMIT = "100/minute"
```

## 🐳 Docker Configuration

### docker-compose.yml

```yaml
services:
  api:
    build: ./python
    ports:
      - "8001:8001"
    environment:
      - OLLAMA_PATH_API=http://ollama:11434/api
    
  mlx:
    build: ./python/mlx_server
    ports:
      - "8003:8003"
    volumes:
      - ./models:/app/models
```

## 🚦 Mode Configuration

### Switching Modes

```bash
# Automatic (via scripts)
./start_hybrid.sh    # Sets AUTOCOMPLETE_MODE=hybrid
./start_trained.sh   # Sets AUTOCOMPLETE_MODE=trained

# Manual override
export AUTOCOMPLETE_MODE=trained
npm run dev
```

### Mode-Specific Settings

**Hybrid Mode:**
- Uses ChromaDB for context
- Calls Ollama for generation
- Higher quality, slightly slower

**Trained Mode:**
- Direct model inference
- No context lookup
- Faster, domain-specific

## 📝 Training Configuration

### Data Preparation

```bash
# prepare_hq_data_fast.sh settings
MIN_WORDS_PER_SENTENCE=8
MAX_WORDS_PER_PROMPT=500
GRAMMAR_CHECK_ENABLED=true
TRAIN_SPLIT=0.8
VALID_SPLIT=0.1
TEST_SPLIT=0.1
```

### Training Parameters

```bash
# start_training_mlx_community.sh
LEARNING_RATE=1e-5
NUM_ITERATIONS=2000
LORA_LAYERS=24
BATCH_SIZE=4
SAVE_EVERY=100
```

## 💾 File Locations

### Configuration Files
- `.env.local` - Environment variables
- `package.json` - Node dependencies
- `requirements.txt` - Python dependencies
- `tsconfig.json` - TypeScript settings

### Data Files
- `/data/bio.json` - Training data
- `/data/custom_dictionary.json` - User dictionary
- `/python/vector_db/chroma_db/` - Vector database

### Log Files
- `/tmp/api_server.log` - API server logs
- `/tmp/mlx_server.log` - MLX server logs
- Chrome DevTools - Frontend logs

## 🔄 Default Values

If not specified, these defaults apply:

```typescript
// Frontend
DEBOUNCE_MS: 1500
MAX_SUGGESTIONS: 3
CACHE_TTL: 300000 (5 min)

// Backend
TEMPERATURE: 0.85
MAX_TOKENS: 25
PORT: 8001 (API), 8003 (MLX)

// Training
LEARNING_RATE: 1e-5
ITERATIONS: 2000
```

## 💡 Configuration Tips

1. **Development**: Use lower debounce for faster testing
2. **Production**: Enable caching, increase rate limits
3. **Performance**: Tune based on hardware capabilities
4. **Quality**: Adjust temperature for creativity vs consistency

Remember: Start with defaults, measure performance, then tune!