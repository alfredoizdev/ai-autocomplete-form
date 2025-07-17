# CLAUDE.md

## Standard Workflow

1. First think through the problem, read the codebase for relevant files, and write a plan to tasks/todo.md.
2. The plan should have a list of todo items that you can check off as you complete them
3. Before you begin working, check in with me and I will verify the plan.
4. Then, begin working on the todo items, marking them as complete as you go.
5. Please every step of the way just give me a high level explanation of what changes you made
6. Make every task and code change you do as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.
7. Finally, add a review section to the todo.md file with a summary of the changes you made and any other relevant information.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sophisticated Next.js 15 application with React 19 that provides AI-powered text autocomplete functionality for personal bio completion, specifically tailored for the swinger community. The app features a hybrid approach combining vector database search (ChromaDB) with LLM generation (Ollama Gemma 3 12B), advanced spell checking, kick.com link detection, and optional fine-tuned model support.

## Key Commands

### Development

- `npm run dev` - Start development server with Turbopack (runs on http://localhost:3000)
- `npm run build` - Build for production (zero errors)
- `npm run start` - Start production server
- `npm run lint` - Run ESLint checks

### Quick Start

```bash
# Start all backend services
./start_all_servers.sh

# Then start Next.js
npm run dev
```

### External Services

- **Ollama**: Must be running locally on port 11434
  - Start: `ollama serve`
  - Pull model: `ollama pull gemma3:12b`
  - Verify: `ollama list`
- **Python API Server**: Port 8001 (hybrid autocomplete)
  - Start: `./start_api_server.sh`
- **Trained Model Server**: Port 8002 (fine-tuned models, optional)
  - Start: `cd python && python -m uvicorn api.trained_model_server:app --port 8002`
- **Docker services** (optional): `docker-compose up -d` (Weaviate + transformers)

## Architecture Overview

### Core Technologies

- **Next.js 15.3.3** with App Router and Server Actions
- **React 19** with TypeScript and advanced hooks
- **Tailwind CSS v4** with PostCSS
- **React Hook Form 7.58.0** for form management
- **typo-js 1.2.5** for spell checking
- **FastAPI** for Python backend services
- **ChromaDB 0.4.24** for vector search
- **Transformers/PyTorch** for model training

### Key Files and Patterns

**AI Integration:**

- `actions/ai-text.ts` - Hybrid API integration (Python server + Ollama fallback)
- `actions/ai-text-streaming.ts` - Streaming responses with smart caching
- `actions/ai-vision.ts` - Image analysis capabilities

**Form Components:**

- `components/Form.tsx` - Main form with all features
- `components/SpellCheckPopup.tsx` & `SpellCheckOverlay.tsx` - Spell checking UI
- `components/KickDetectionWarning.tsx` - Content filtering warnings

**Hook Architecture (5-hook system):**

- `hooks/useFormAutocomplete.tsx` - Core autocomplete logic
- `hooks/useSpellCheck.tsx` - Spell checking with contractions
- `hooks/useDebouncedSpellCheck.tsx` - Performance wrapper
- `hooks/useTextFeatureCoordinator.tsx` - Feature conflict prevention
- `hooks/useKickDetection.tsx` - Pattern matching for prohibited content

**Backend Services:**

- `python/api/api_server.py` - Main API server (port 8001)
- `python/api/trained_model_server.py` - Fine-tuned models (port 8002)
- `python/vector_db/` - ChromaDB vector search implementation
- `python/mlx_training/` - Model training scripts

### Environment Configuration

The app requires `.env.local` with:

```
OLLAMA_PATH_API=http://127.0.0.1:11434/api
# Optional: Enable fine-tuned model integration
NEXT_PUBLIC_USE_FINETUNED_MODEL=true
# Optional: OpenAI API key for fallback
NEXT_PUBLIC_OPENAI_API_KEY=your-key-here
```

### Project Structure

```
actions/        # Server actions for AI integration
├── ai-text.ts              # Hybrid API integration
├── ai-text-streaming.ts    # Streaming with caching
└── ai-vision.ts            # Image analysis

app/           # Next.js app router pages and layouts
├── page.tsx                # Main bio autocomplete
├── optimized/              # Performance demo route
├── ai-image/               # Image analysis feature
├── test-kick/              # Kick detection testing
└── api/                    # API routes

components/    # React components
├── Form.tsx                # Full-featured form
├── SpellCheckPopup.tsx     # Spell suggestions
└── KickDetectionWarning.tsx # Safety warnings

hooks/         # Sophisticated 5-hook architecture
├── useFormAutocomplete.tsx # Main autocomplete
├── useSpellCheck.tsx       # Spell checking
└── useTextFeatureCoordinator.tsx # Conflict prevention

lib/           # Core utilities
├── kickDetection.ts        # 40+ pattern matching
├── customDictionary.ts     # Persistent dictionary
└── utils.ts                # Helper functions

python/        # Backend services
├── api/                    # FastAPI servers
├── vector_db/              # ChromaDB setup
└── mlx_training/           # Model training

data/          # Training data
└── bio.json                # 5000+ bio examples
```

## Important Notes

- No test framework configured - recommend Jest/React Testing Library
- ESLint configured with Next.js recommended rules
- Tailwind CSS v4 with PostCSS (latest version)
- ChromaDB replaces Weaviate for vector search
- Hybrid approach: Python API server preferred, Ollama fallback
- Production build passes with zero TypeScript errors
- Mobile-optimized with 16px fonts and responsive design

## Key Features

### 1. Hybrid AI Autocomplete
- Vector search (ChromaDB) + LLM generation (Ollama)
- 100-150ms response times (60-80% faster with streaming)
- Smart caching with 5-minute TTL
- Adaptive debouncing (50-400ms)
- 3-4 word trigger threshold

### 2. Advanced Spell Checking
- typo-js with Hunspell dictionaries
- 80+ contraction handling
- Custom dictionary with persistence
- Click-to-correct interface
- Performance optimized (800ms debounce)

### 3. Kick.com Detection v2
- 40+ obfuscation patterns
- Phonetic variation detection
- Zero-width character support
- Multi-layer detection approach
- Sub-5ms performance

### 4. Text Feature Coordination
- Prevents conflicts between features
- Adaptive locking system
- Memory management
- Seamless multi-feature operation

### 5. Fine-tuned Models (Optional)
- GPT-2 and DistilGPT2 variants
- Trained on 5000+ bio examples
- Separate server on port 8002
- 80-120ms inference time

## Performance Metrics

- Standard mode: 100-150ms
- Optimized mode: 50-100ms
- Vector search: ~100ms
- LLM generation: 200-500ms
- Fine-tuned models: 80-120ms
- Cache hit rate: 90%
- Kick detection: <5ms

## Standard Commands to Run

When making code changes, always run these commands to ensure quality:

```bash
# Lint check
npm run lint

# Build check (catches TypeScript errors)
npm run build

# Development server
npm run dev
```

## Recent Architecture Updates

The codebase has undergone significant improvements:

1. **Hybrid API Approach**: Python API server is tried first, with Ollama fallback
2. **Streaming Support**: Real-time character-by-character display in optimized route
3. **Smart Caching**: 5-minute TTL cache reduces API calls by 90%
4. **5-Hook Architecture**: Sophisticated system for feature coordination
5. **40+ Kick Patterns**: Enhanced detection with phonetic and zero-width support
6. **Fine-tuned Models**: Optional GPT-2 variants for faster, specialized inference

## Important Workflow Notes

- Always check `tasks/todo.md` for recent changes and architecture updates
- The Python API server (port 8001) is the preferred autocomplete source
- Fine-tuned model server (port 8002) is optional but provides faster inference
- Use `./start_all_servers.sh` for quick backend setup
