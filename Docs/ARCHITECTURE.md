# Architecture Overview

Understanding how the AI Bio Autocomplete system works.

## 🏗️ System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    User Browser                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Next.js Frontend (Port 3000)            │   │
│  │                                                      │   │
│  │  Form ──► Hooks ──► Server Actions ──► API Calls   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend Services                          │
│                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐ │
│  │  Hybrid API    │  │  MLX Server    │  │   Ollama     │ │
│  │  Port 8001     │  │  Port 8003     │  │  Port 11434  │ │
│  │                │  │                │  │              │ │
│  │ Vector Search  │  │ HIGH-QUALITY   │  │ Gemma3 12B   │ │
│  │ + AI Gen       │  │ Llama 3.2-3B   │  │ Generation   │ │
│  └───────┬────────┘  └────────────────┘  └──────┬───────┘ │
│          │                                        │          │
│          ▼                                        │          │
│  ┌────────────────┐                              │          │
│  │   ChromaDB     │◄─────────────────────────────┘          │
│  │ Vector Store   │                                         │
│  │  5000+ Bios    │                                         │
│  └────────────────┘                                         │
│                                                              │
│  Training Data:                                             │
│  • bio.json (5k examples)                                   │
│  • LookingFor_20000.csv (19k examples → 4.5k HQ training)  │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 Request Flow

### Hybrid Mode (Default)
1. User types in textarea
2. Frontend debounces input (1.5s)
3. Server action checks mode
4. Calls Hybrid API on port 8001
5. API performs vector search
6. Generates with Ollama
7. Returns combined results
8. Frontend displays suggestions

### Trained Mode (Fast)
1. User types in textarea
2. Frontend debounces input
3. Server action checks mode
4. Calls MLX server on port 8003
5. Fine-tuned model generates
6. Returns completion
7. Frontend displays result

## 🧩 Key Components

### Frontend Architecture

**5-Hook System:**
```typescript
useFormAutocomplete     // Main form logic
useSpellCheck          // Spell checking
useDebouncedSpellCheck // Performance wrapper
useTextFeatureCoordinator // Prevents conflicts
useKickDetection       // Content filtering
```

**Server Actions:**
- `ai-text.ts` - Mode routing and API calls
- `ai-vision.ts` - Image analysis

### Backend Services

**Hybrid API Server (8001):**
- Vector similarity search
- Context building
- LLM orchestration
- Response filtering

**MLX Model Server (8003):**
- Model loading (prioritizes HIGH-QUALITY model)
- Fast inference:
  - HIGH-QUALITY 3B: 100-150ms (best)
  - Standard 3B: 100-150ms
  - 1B models: 50-100ms
- Grammar correction built-in
- Batch processing support
- Model priority order:
  1. HIGH-QUALITY Llama-3.2-3B (2000 iterations, grammar-filtered)
  2. Standard Llama-3.2-3B (1500 iterations)
  3. Llama-3.2-1B (1000 iterations, faster)
  4. Legacy bio models

## 📊 Data Flow

### Text Processing Pipeline
```
Input Text
    ↓
Debouncing (1.5s)
    ↓
Feature Coordinator
    ├─► Spell Check
    ├─► Kick Detection
    └─► Autocomplete
         ↓
    Server Action
         ↓
    [Mode Check]
    /          \
Hybrid      Trained
   ↓            ↓
Vector+AI   MLX Model
   ↓            ↓
Suggestions  Completion
```

### Caching Strategy
- Frontend: 5-minute response cache
- Vector DB: Persistent embeddings
- Model: Loaded in memory

## 🔐 Security Features

1. **Content Filtering**
   - 40+ kick.com patterns
   - Real-time detection
   - Zero-width character support

2. **Input Validation**
   - Length limits
   - Character filtering
   - Rate limiting ready

3. **CORS Protection**
   - Localhost only (dev)
   - Configurable for production

## 🚀 Performance Optimizations

### Frontend
- Progressive debouncing
- Smart caching
- Lazy component loading
- Optimistic UI updates

### Backend
- Connection pooling
- Model preloading
- Vector index optimization
- Async request handling

### Response Times
```
Operation          Target    Actual
─────────────────────────────────
Vector Search      <100ms    ~80ms
AI Generation      <500ms    ~350ms
Hybrid Total       <200ms    ~150ms
MLX HQ 3B         <150ms    ~120ms
MLX Std 3B        <150ms    ~130ms
MLX 1B            <100ms    ~70ms
Spell Check        <50ms     ~20ms
Kick Detection     <10ms     ~3ms
```

## 🔧 Configuration Points

### Environment Variables
```bash
OLLAMA_PATH_API        # Ollama endpoint
AUTOCOMPLETE_MODE      # hybrid or trained
```

### Tuning Parameters
```javascript
// Frontend
DEBOUNCE_DELAY = 1500
MIN_WORDS = 5
CACHE_DURATION = 300000

// Backend
NUM_SIMILAR_BIOS = 10
MIN_SUGGESTION_LENGTH = 8
TEMPERATURES = [0.7, 0.9]
```

## 🏭 Production Considerations

### Scaling
- Stateless API servers
- Model server pooling
- CDN for static assets
- Database replication

### Monitoring
- Response time tracking
- Error rate monitoring
- Model performance metrics
- Resource utilization

### Deployment
```
Load Balancer
      ↓
┌─────┴─────┐
│  Frontend │ (Multiple instances)
└─────┬─────┘
      ↓
┌─────┴─────┐
│    API    │ (Horizontal scaling)
└─────┬─────┘
      ↓
┌─────┴─────┐
│  Services │ (Ollama, MLX, DB)
└───────────┘
```

## 💡 Design Decisions

### Why Hybrid Architecture?
- Best of both worlds
- Fallback options
- Progressive enhancement
- User choice

### Why 5-Hook System?
- Separation of concerns
- Reusability
- Testability
- Performance isolation

### Why MLX for Training?
- Apple Silicon optimization
- Memory efficiency with LoRA
- Fast inference (100-150ms)
- Easy deployment
- Grammar-filtered training data
- 2000 iterations for quality

## 🔮 Future Architecture

Planned improvements:
- WebSocket for real-time
- Multi-model ensemble
- Edge deployment
- Federated learning

This architecture prioritizes:
1. **User Experience** - Fast, reliable responses
2. **Developer Experience** - Clear, modular code
3. **Operational Excellence** - Easy to deploy and monitor