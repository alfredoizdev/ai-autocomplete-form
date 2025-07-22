# Hybrid Mode Operation Guide

This guide explains how the hybrid mode works and how to operate it effectively. Hybrid mode combines vector search with AI generation for the best balance of speed and quality.

## 🎯 What is Hybrid Mode?

Hybrid mode uses two technologies together:
1. **Vector Search** (ChromaDB) - Finds similar existing bios instantly
2. **AI Generation** (Ollama) - Creates new text based on context

Think of it like having a smart assistant that:
- First looks through thousands of examples to find similar ones
- Then uses those examples to help write something new and unique

## 🏗️ How Hybrid Mode Works

### The Complete Flow

```
User Types → Frontend → API Server → Vector Search + AI Generation → Response
```

### Step-by-Step Process

1. **User Input** (Frontend)
   - User types at least 5 words
   - System waits for a pause in typing (1.5 seconds)
   - Text is sent to the API server

2. **Vector Search** (ChromaDB)
   - Converts user's text to mathematical vectors
   - Searches 5000+ bio examples for similar ones
   - Returns top 10 most similar bios
   - Time: ~50-100ms

3. **Context Building** (API Server)
   - Takes the similar bios from vector search
   - Creates a context prompt for the AI
   - Adds user's partial text as the prompt

4. **AI Generation** (Ollama)
   - Receives context + prompt
   - Generates 2 different completions
   - Uses different "temperatures" for variety
   - Time: ~200-500ms

5. **Response Processing**
   - Combines vector search results and AI generations
   - Filters out incomplete sentences
   - Ranks by quality and relevance
   - Returns top 3 suggestions

## 🚀 Starting Hybrid Mode

### Method 1: Automatic Startup (Recommended)
```bash
# This starts everything you need
./start_all_servers.sh

# Then in a new terminal
npm run dev
```

### Method 2: Manual Startup
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start API Server
cd python
source venv/bin/activate
python api/api_server.py

# Terminal 3: Start Frontend
npm run dev
```

## 🔍 Monitoring Hybrid Mode

### Check Service Status
```bash
# Run the status checker
python python/check_status.py
```

You should see:
```
✅ Ollama is running
✅ API Server is running on port 8001
✅ ChromaDB database exists
   Indexed bios: 5000
```

### View Real-time Logs

**API Server Logs:**
```bash
tail -f python/api_server.log
```

**See API Requests:**
```bash
# In the terminal running the API server
# You'll see each request and response time
```

### API Documentation
Visit: http://localhost:8001/docs

This shows:
- All available endpoints
- Request/response formats
- Interactive testing interface

## ⚙️ Configuring Hybrid Mode

### Environment Variables (.env.local)
```bash
# Required
OLLAMA_PATH_API=http://127.0.0.1:11434/api

# Optional
NEXT_PUBLIC_USE_FINETUNED_MODEL=false  # Set to true for MLX model
```

### API Server Settings

Edit `python/api/api_server.py` to adjust:

```python
# Number of similar bios to find
NUM_SIMILAR_BIOS = 10

# Minimum completion length
MIN_COMPLETION_LENGTH = 8

# AI generation temperature
TEMPERATURES = [0.7, 0.9]  # Lower = more focused, Higher = more creative
```

### Performance Tuning

**Frontend Debouncing** (in `hooks/useFormAutocomplete.tsx`):
```javascript
// Adjust typing delay before autocomplete
const DEBOUNCE_DELAY = 1500;  // milliseconds

// Words needed to trigger
const MIN_WORDS = 5;
```

**Cache Duration** (in `actions/ai-text-streaming.ts`):
```javascript
// How long to cache suggestions
const CACHE_DURATION = 5 * 60 * 1000;  // 5 minutes
```

## 📊 Understanding the API Endpoints

### Main Endpoints

1. **GET /api/stats**
   - Shows database statistics
   - Number of indexed bios

2. **POST /api/autocomplete**
   - Vector search only (fast)
   - Returns exact matches

3. **POST /api/autocomplete/hybrid**
   - Full hybrid mode
   - Vector search + AI generation

### Testing Endpoints

```bash
# Test vector search only
curl -X POST http://localhost:8001/api/autocomplete \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I am a fun loving person who"}'

# Test hybrid mode
curl -X POST http://localhost:8001/api/autocomplete/hybrid \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I am a fun loving person who"}'
```

## 🎯 Optimizing Results

### Better Vector Search Results

1. **Index Quality**: The vector database uses 5000+ real bio examples
2. **Embedding Model**: Uses sentence-transformers for semantic understanding
3. **Similarity Threshold**: Adjust in `vector_search.py`:
   ```python
   results = collection.query(
       query_embeddings=[embedding],
       n_results=10  # Increase for more variety
   )
   ```

### Better AI Generation

1. **Context Window**: More similar bios = better context
2. **Temperature Settings**: 
   - 0.5-0.7: More predictable, safer
   - 0.8-1.0: More creative, riskier
3. **Prompt Engineering**: The system adds hidden prompts for quality

## 🔧 Troubleshooting Hybrid Mode

### Slow Response Times

1. **Check Ollama Model**:
   ```bash
   # Ensure model is loaded
   ollama list
   
   # Pre-load model
   ollama run gemma3:12b "test"
   ```

2. **Monitor API Performance**:
   - Check `api_server.log` for timing
   - Vector search should be <100ms
   - AI generation typically 200-500ms

### Poor Quality Suggestions

1. **Verify Database**:
   ```bash
   # Re-index if needed
   cd python
   python vector_db/setup_chromadb.py
   ```

2. **Check Context Building**:
   - Look at API logs to see what context is being sent
   - Ensure similar bios are actually relevant

### Connection Issues

```bash
# Test each component
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:8001/api/stats  # API Server
```

## 📈 Performance Metrics

### Expected Performance
- **Vector Search**: 50-100ms
- **AI Generation**: 200-500ms  
- **Total Response**: 300-600ms (first request)
- **Cached Response**: <50ms

### Memory Usage
- **Ollama**: 8-10GB (model loaded)
- **API Server**: 200-500MB
- **ChromaDB**: 100-200MB
- **Frontend**: 100-200MB

## 🎮 Advanced Features

### 1. Streaming Responses
The system supports real-time streaming:
- Characters appear as they're generated
- Better perceived performance
- Enabled automatically for longer responses

### 2. Smart Caching
- Recent suggestions cached for 5 minutes
- Identical prompts return instantly
- Cache cleared on text changes

### 3. Fallback Mechanism
If vector search fails, system falls back to:
1. Pure AI generation
2. OpenAI API (if configured)
3. Error message with helpful context

## 🎯 Best Practices

1. **Keep Services Running**: Don't close terminal windows
2. **Monitor Logs**: Watch for errors or slow queries  
3. **Regular Maintenance**: Restart services daily for best performance
4. **Update Models**: Check for Ollama model updates

## 📚 Next Steps

- Learn about [Training Custom Models](./LOCAL_LLM_TRAINING_GUIDE.md)
- Explore [Advanced Configuration](./ADVANCED_CONFIG.md)
- Read about [Architecture Details](./ARCHITECTURE.md)

---

Hybrid mode gives you the best of both worlds: the speed of vector search with the creativity of AI generation. With this guide, you can operate and optimize it effectively!