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

### Quick Start - Choose Your Mode

#### Option 1: Hybrid Mode (Vector Search + AI)

```bash
# Uses ChromaDB vector search + Ollama for best quality
./start_hybrid.sh

# Then in a new terminal
npm run dev
```

#### Option 2: Trained Model Mode (Fine-tuned Llama 3.2)

```bash
# Uses your locally trained MLX model for fastest speed
./start_trained.sh

# Then in a new terminal
npm run dev
```

#### Option 3: Train Your Own Model

```bash
# Prepare high-quality data and start training
./prepare_hq_data_fast.sh  # Prepares grammar-filtered high-quality data
./start_training_mlx_community.sh  # Trains with optimized parameters
```

### Training Data Format Requirements

When preparing new training data for fine-tuning, follow these specific requirements:

#### Data Format

- **Input format**: CSV file with bio text in the first column
- **Output format**: JSONL files with prompt-completion pairs
- **Each training example must be exactly ONE complete sentence**
- **Multi-sentence bios are split into separate training examples**

#### Quality Requirements

1. **Minimum length**: 8 words per sentence (shorter sentences are discarded)
2. **Maximum length**: 500 words per prompt-completion pair
3. **Sentence structure**:
   - Each prompt must NOT end with punctuation (.!?)
   - Each completion MUST end with proper punctuation
   - Combined prompt + completion forms one grammatically correct sentence

#### Processing Steps

1. **Load CSV data** - Handle quotes and encoding issues
2. **Split into sentences** - Each bio is split into individual sentences
3. **Create smart splits** - Find natural break points within each sentence:
   - After conjunctions (and, but, or)
   - Before relative pronouns (who, which, that)
   - After key phrases ("looking for", "interested in", "seeking")
   - At commas in appropriate positions
4. **Validate quality** - Ensure all requirements are met
5. **Create train/valid/test splits** - 80%/10%/10% ratio

#### To Process New Training Data

```bash
# 1. Place your CSV file in data/ directory (e.g., data/new_data.csv)

# 2. Use the high-quality data preparation script:
./prepare_hq_data_fast.sh

# 3. This script will:
#    - Load your CSV data
#    - Apply grammar filtering
#    - Create natural split points
#    - Generate high-quality prompt-completion pairs
#    - Output to python/mlx_training/lookingfor_hq/

# 4. Train the model:
./start_training_mlx_community.sh
```

#### Example Training Data Format

After processing, each JSONL line contains:

```json
{
  "text": "<|user|>\nComplete this bio: Looking for fun loving people<|end|>\n<|assistant|>\nthat we can have fun with in and out of the bedroom.<|end|>"
}
```

This creates a natural sentence completion task where the model learns to complete partial bio sentences in a coherent way.

### External Services

- **Ollama**: Must be running locally on port 11434
  - Start: `ollama serve`
  - Pull model: `ollama pull gemma3:12b`
  - Verify: `ollama list`
- **Python API Server**: Port 8001 (hybrid autocomplete)
  - Start: `./start_hybrid.sh` (recommended)
- **MLX Model Server**: Port 8003 (fine-tuned Llama models)
  - Start: `./start_trained.sh` (recommended) or manually
  - Models (in priority order):
    - HIGH-QUALITY Llama-3.2-3B-Instruct fine-tuned on grammar-filtered LookingFor dataset (default - best quality, 2000 iterations)
    - Llama-3.2-3B-Instruct fine-tuned on LookingFor dataset (1500 iterations)
    - Llama-3.2-1B-Instruct fine-tuned on LookingFor dataset (faster alternative)
    - Llama-3.2-3B-Instruct fine-tuned on bio dataset
    - Legacy Phi-3 models
- **Docker services** (optional): `docker-compose up -d` (legacy Weaviate)

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
- **MLX** for Apple Silicon optimized training and inference

### Key Files and Patterns

**AI Integration:**

- `actions/ai-text.ts` - Hybrid API integration (Python server + Ollama fallback)
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
- `python/vector_db/` - ChromaDB vector search implementation
- `python/mlx_training/` - Model training scripts
- `python/mlx_server/mlx_model_server.py` - MLX server for Llama models (port 8003)

### Environment Configuration

The app requires `.env.local` with:

```
OLLAMA_PATH_API=http://127.0.0.1:11434/api
# Set the autocomplete mode: 'hybrid' or 'trained'
AUTOCOMPLETE_MODE=hybrid
# Optional: OpenAI API key for fallback
NEXT_PUBLIC_OPENAI_API_KEY=your-key-here
```

Note: The mode is automatically set by the startup scripts (`start_hybrid.sh` or `start_trained.sh`)

### Project Structure

```
actions/        # Server actions for AI integration
├── ai-text.ts              # Hybrid API integration
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
├── bio.json                # 5000+ bio examples
└── LookingFor_20000.csv    # 19k+ bio examples (LookingFor dataset)
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
- 100-150ms response times
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

- HIGH-QUALITY Llama-3.2-3B-Instruct with LoRA adapters (default - grammar-filtered LookingFor dataset, 2000 iterations)
- Llama-3.2-3B-Instruct with LoRA adapters (LookingFor dataset, 1500 iterations)
- Llama-3.2-1B-Instruct with LoRA adapters (LookingFor dataset, faster option)
- Llama-3.2-3B-Instruct with LoRA adapters (bio dataset)
- Trained on high-quality sentence-based bio data with grammar validation
- MLX server on port 8003 (Apple Silicon optimized)
- 100-150ms inference time (3B model) / 50-100ms (1B model)

## Performance Metrics

- Standard mode: 100-150ms
- Optimized mode: 50-100ms
- Vector search: ~100ms
- LLM generation: 200-500ms
- Fine-tuned models: 50-100ms (1B) / 100-150ms (3B)
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

1. **Mode-based Architecture**: Switch between 'hybrid' and 'trained' modes via environment variable
2. **MLX Training Support**: Train Llama-3.2 models locally on Apple Silicon
3. **Grammar-Filtered Training**: New high-quality dataset with grammar validation (4.5k+ examples)
4. **Improved Shell Scripts**: Only 4 essential scripts: `start_hybrid.sh`, `start_trained.sh`, `prepare_hq_data_fast.sh`, `start_training_mlx_community.sh`
5. **Sentence-based Training Data**: Higher quality bio completions with natural sentence structure
6. **Smart Caching**: 5-minute TTL cache reduces API calls by 90%
7. **5-Hook Architecture**: Sophisticated system for feature coordination
8. **40+ Kick Patterns**: Enhanced detection with phonetic and zero-width support
9. **LookingFor Dataset**: Grammar-filtered training examples with natural split points
10. **Multiple Model Support**: MLX server prioritizes models: HIGH-QUALITY 3B → Standard 3B → 1B → bio → Phi-3
11. **Enhanced Training**: 2000 iterations with optimized learning rate (1e-5) for superior quality

## Important Workflow Notes

- Always check `tasks/todo.md` for recent changes and architecture updates
- Choose your mode with startup scripts:
  - `./start_hybrid.sh` - Best quality with vector search + AI
  - `./start_trained.sh` - Fastest speed with fine-tuned model
- The Python API server (port 8001) handles hybrid mode
- MLX model server (port 8003) handles trained mode with Llama models

## Documentation

Comprehensive documentation has been created to help developers understand and run this project:

### Documentation Structure

All documentation is located in the `Docs/` folder:

- **[Getting Started](./Docs/GETTING_STARTED.md)** - Complete setup guide
- **[Running the App](./Docs/RUNNING_THE_APP.md)** - Hybrid and trained modes explained
- **[Training Guide](./Docs/TRAINING_GUIDE.md)** - Train your own HIGH-QUALITY model
- **[API Reference](./Docs/API_REFERENCE.md)** - Backend API documentation
- **[Architecture](./Docs/ARCHITECTURE.md)** - System design and data flow
- **[Troubleshooting](./Docs/TROUBLESHOOTING.md)** - Common issues and solutions

### Quick Links

1. First time? Start here: [Getting Started](./Docs/GETTING_STARTED.md)
2. Want to train a model? See: [Training Guide](./Docs/TRAINING_GUIDE.md)
3. Having issues? Check: [Troubleshooting](./Docs/TROUBLESHOOTING.md)
4. Need API details? Read: [API Reference](./Docs/API_REFERENCE.md)

## Latest Updates (July 23, 2025)

### Grammar-Filtered Training Data
- Created `prepare_hq_data_fast.sh` for high-quality data preparation
- Filters training data for grammatical correctness
- Reduces dataset from 15k to 4.5k high-quality examples
- Uses natural split points for better sentence completion

### Optimized Training Process
- `start_training_mlx_community.sh` with optimized parameters:
  - Learning rate: 1e-5 (reduced from 5e-5)
  - Iterations: 2000 (increased from 1500)
  - Layers: 24 (increased from 16)
  - Uses MLX community models (no authentication required)

### Shell Script Cleanup
- Removed 6 obsolete scripts
- Kept only 4 essential scripts:
  - `start_hybrid.sh` - Hybrid mode with vector search
  - `start_trained.sh` - Trained model mode
  - `prepare_hq_data_fast.sh` - Data preparation
  - `start_training_mlx_community.sh` - Model training

### Model Priority System
- MLX server now prioritizes HIGH-QUALITY model first
- Located at: `models/lookingfor-llama3-3b-hq-lora/`
- Trained with grammar-filtered dataset for superior output quality

## Additional Development Tools

### Python Environment Setup
- Virtual environment located at `python/venv/`
- Two requirements files:
  - `python/requirements.txt` - Main dependencies (ChromaDB, FastAPI, MLX)
  - `python/mlx_server/requirements.txt` - MLX server specific dependencies
- Hugging Face token required for model downloads (see `python/.env.example`)

### Status Check Utility
Run `python python/check_status.py` to verify:
- Service availability (API servers, Ollama, Docker)
- Database existence and stats
- MLX installation status
- Training data preparation
- Quick start commands

### Vector Database Utilities
Located in `python/vector_db/`:
- `setup_chromadb.py` / `setup_chromadb_improved.py` - Database initialization
- `check_chromadb_status.py` - Verify database health
- `analyze_bio_lengths.py` - Analyze training data statistics
- `clean_bio_data.py` - Data preprocessing utility
- `vector_search.py` - Direct vector search testing

### Training Data
Multiple datasets available:
- `data/bio.json` - 5000+ examples
- `data/lookingfor_20000.json` - Converted from CSV
- `data/newBios20000.csv` - Original CSV data
- `data/bio_cleaned.json` - Preprocessed version
- Conversion utility: `convert_csv_to_json.py` (root directory)

### Model Storage Structure
Trained models in `models/`:
- `lookingfor-llama3-3b-hq-lora/` - HIGH-QUALITY grammar-filtered model
- `lookingfor-llama3-3b-lora/` - Standard model with checkpoint history
- `bio-sentence-llama3-lora/` - Bio dataset trained model
- Each model includes checkpoints saved during training

### MLX Configuration
Default config at `python/mlx_server/config.yaml`:
- Uses Phi-3-mini-4k-instruct-4bit by default
- Optimized for M1 Max with 32GB RAM
- LoRA rank 16, alpha 32 for memory efficiency
- Batch size 2 with gradient accumulation
- Mixed precision and gradient checkpointing enabled