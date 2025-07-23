# AI Bio Autocomplete with Hybrid Vector Search

A sophisticated AI-powered bio autocomplete system built with Next.js 15, React 19, and Python FastAPI. Combines vector database search (ChromaDB) with LLM generation (Ollama Gemma 3 12B) for high-quality, contextually relevant bio completions.

## 🚀 Key Features

- **Hybrid AI Autocomplete** - Vector search + LLM generation for 100-150ms response times
- **Advanced Spell Checking** - typo-js with Hunspell dictionaries, custom dictionary, and 80+ contraction support
- **Kick.com Detection v2** - 40+ obfuscation patterns including phonetic and zero-width character detection
- **Smart Feature Coordination** - Prevents conflicts between autocomplete, spell check, and other features
- **Mobile-Optimized** - 16px fonts, responsive design, and touch-friendly interface
- **Streaming Responses** - Character-by-character display for 60-80% faster perceived latency
- **Fine-tuned Models** - Optional Llama-3.2 models (3B/1B) trained on 15k+ bio examples

## Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js** (version 18 or higher)
- **Python** (version 3.8 or higher)
- **npm** or **yarn**
- **Ollama** (for running the Gemma 3 12B model locally)
- **Git** (for cloning the repository)
- **PyTorch** (for running fine-tuned models - optional)

## Ollama Setup

1. **Install Ollama** on your system:

   - Visit [Ollama's official website](https://ollama.ai) and download the installer for your operating system
   - Follow the installation instructions

2. **Pull the Gemma 3 12B model**:

   ```bash
   ollama pull gemma3:12b
   ```

3. **Start Ollama service**:

   ```bash
   ollama serve
   ```

4. **Verify the installation**:
   ```bash
   ollama list
   ```
   You should see `gemma3:12b` in the list of available models.

## Installation

1. **Clone and install dependencies**:
   ```bash
   git clone <your-repository-url>
   cd ai-train-llm
   npm install
   ```

2. **Set up Python environment**:
   ```bash
   cd python
   python3 -m venv venv
   source venv/bin/activate  # Mac/Linux (or venv\Scripts\activate on Windows)
   pip install -r requirements.txt
   cd ..
   ```

3. **Configure environment** (`.env.local`):
   ```env
   OLLAMA_PATH_API=http://127.0.0.1:11434/api
   NEXT_PUBLIC_USE_FINETUNED_MODEL=true  # Optional
   ```

4. **Initialize vector database**:
   ```bash
   cd python/vector_db && python setup_chromadb.py && cd ../..
   ```

## Running the Full Stack

### Quick Start - Choose Your Mode

#### Option A: Hybrid Mode (Vector Search + AI)
```bash
./start_hybrid.sh
# Then in another terminal:
npm run dev
```

#### Option B: Trained Model Mode (Fine-tuned Llama)
```bash
./start_trained.sh
# Then in another terminal:
npm run dev
```

#### Option C: Start All Services
```bash
./start_all_servers.sh
# Then in another terminal:
npm run dev
```

### Manual Start (Individual Services)

#### 1. **Start Ollama** (Terminal 1):
```bash
ollama serve
```

#### 2. **Start Python API Server** (Terminal 2):
```bash
./start_api_server.sh
# or manually:
cd python && python api/api_server.py
```
The API server will run on `http://localhost:8001`

#### 3. **Start MLX Model Server** (Terminal 3 - Optional):
```bash
cd python/mlx_server
python mlx_model_server.py
```
The MLX server will run on `http://localhost:8003` with Llama-3.2 models

#### 4. **Start Next.js Development Server** (Terminal 4):
```bash
npm run dev
# or
yarn dev
```
The web app will run on `http://localhost:3000`

## Available Routes

- **`/` (Main Application)** - Full-featured bio autocomplete with all capabilities
- **`/ai-image`** - Image analysis feature for bio photos
- **`/test-kick`** - Visual test page for kick detection patterns (dev only)

## Usage

Navigate to `http://localhost:3000` and start typing in the bio field:

- **AI suggestions** appear after 3-5 words (press Tab to accept)
- **Click** misspelled words for corrections
- **Kick.com detection** shows real-time warnings
- All features work seamlessly together on desktop and mobile

## Python API Server

The application includes a FastAPI server that provides the hybrid autocomplete functionality:

### API Endpoints

#### Main API Server (Port 8001)
- **GET /** - Health check endpoint
- **POST /api/autocomplete** - Vector-only autocomplete suggestions
- **POST /api/autocomplete/hybrid** - Hybrid autocomplete (vector + LLM)
- **GET /api/stats** - Database statistics

### Hybrid Approach

The hybrid autocomplete system combines:
1. **Vector Search** - Fast exact matches from ~5000 bio examples using ChromaDB
2. **LLM Generation** - Creative completions using Ollama Gemma 3 12B
3. **Quality Filtering** - Ensures suggestions are complete thoughts (8+ words)

### Performance Metrics

- **Response Time**: 100-150ms (hybrid mode)
- **Vector Search**: ~100ms
- **LLM Generation**: 200-500ms
- **Fine-tuned Llama Model**: 50-150ms
- **Kick Detection**: <5ms
- **Cache Hit Rate**: 90%

### API Documentation

- **Main API Server**: `http://localhost:8001/docs`
- **MLX Model Server**: `http://localhost:8003/docs` (when running)

Both servers provide interactive Swagger/OpenAPI documentation.

## How It Works

### System Architecture

```
User Input → Next.js Form → Multiple Processing Layers
            ↓                           ↓
    ┌───────────────┐        ┌──────────────────┐
    │ Kick Detection│        │ Python API (8001)│
    │ Pattern Match │        │                  │
    │ Fuzzy Logic   │        ├──────────────────┤
    │ Homoglyphs    │        │ Vector Search    │
    └───────┬───────┘        │ (ChromaDB)       │
            ↓                │ ~5000 Bios       │
    ┌───────────────┐        └────────┬─────────┘
    │ Warning UI    │                 ↓
    │ Confidence    │        ┌──────────────────┐
    │ Logging       │        │ LLM Generation   │
    └───────────────┘        │ (Ollama Gemma 3) │
                             │ Streaming/Batch  │
                             └────────┬─────────┘
                                      ↓
                             ┌──────────────────┐
                             │ Smart Filter     │
                             │ Quality Check    │
                             │ Top 3 Results    │
                             └──────────────────┘
```

### Technical Implementation

The application uses a sophisticated 5-hook architecture for optimal performance:

1. **`useFormAutocomplete`** - Main form logic with AI integration
2. **`useSpellCheck`** - Core spell checking with typo-js
3. **`useDebouncedSpellCheck`** - Performance-optimized wrapper
4. **`useTextFeatureCoordinator`** - Prevents feature conflicts
5. **`useKickDetection`** - Real-time pattern matching

**Text Feature Coordinator** manages three features (AUTOCOMPLETE, SPELLCHECK, CAPITALIZATION) with:
- Feature locking with adaptive durations
- Coexistence support for simultaneous operations
- Automatic memory cleanup
- Conflict prevention between features

## Project Structure

```
ai-train-llm/
├── actions/
│   ├── ai-text.ts                    # Server actions for hybrid API integration
│   ├── ai-text-streaming.ts          # Streaming responses with smart caching
│   └── ai-vision.ts                  # Image analysis actions
├── app/
│   ├── layout.tsx                    # Root layout
│   ├── page.tsx                      # Main page component
│   ├── api/
│   │   └── kick-detection-logs/      # Logging endpoint for pattern learning
│   ├── ai-image/                     # Image analysis feature
│   └── globals.css                   # Global styles with Tailwind v4
├── components/
│   ├── Form.tsx                      # Main form with layered textarea
│   ├── FormImage.tsx                 # Image upload form
│   ├── SpellCheckPopup.tsx           # Interactive spell suggestion popup
│   ├── SpellCheckOverlay.tsx         # Spell check visual overlay
│   ├── KickDetectionWarning.tsx      # Warning UI for kick.com detection
│   └── Navbar.tsx                    # Navigation component
├── hooks/                            # Sophisticated hook architecture
│   ├── useFormAutocomplete.tsx       # Main form logic with AI integration
│   ├── useSpellCheck.tsx             # Core spell check with contractions
│   ├── useDebouncedSpellCheck.tsx    # Performance-optimized wrapper
│   ├── useTextFeatureCoordinator.tsx # Feature conflict prevention
│   └── useKickDetection.tsx          # Real-time pattern matching
├── lib/
│   ├── customDictionary.ts           # Custom dictionary service
│   ├── kickDetection.ts              # Kick.com pattern matching engine
│   ├── openai.ts                     # OpenAI integration (unused)
│   └── utils.ts                      # Utility functions
├── data/
│   └── bio.json                      # ~5000 bio examples for vector database
├── python/                           # Python backend
│   ├── api/
│   │   └── api_server.py            # FastAPI hybrid autocomplete server
│   ├── vector_db/
│   │   ├── setup_chromadb.py        # Initialize vector database
│   │   └── vector_search.py         # Vector search implementation
│   ├── mlx_training/                # Model training scripts
│   │   └── bio_dataset/             # Training datasets
│   ├── chroma_db/                   # ChromaDB persistent storage
│   ├── requirements.txt             # Python dependencies
│   ├── setup.sh                     # Environment setup script
│   └── activate.sh                  # Virtual env activation helper
├── public/
│   ├── dictionaries/
│   │   └── en_US/                   # Hunspell dictionaries
│   │       ├── en_US.aff           # Affix rules
│   │       └── en_US.dic           # Dictionary words
│   └── images/                      # Static images
├── type/                            # TypeScript type definitions
├── [Configuration Files]
│   ├── next.config.ts               # Next.js configuration
│   ├── tailwind.config.js           # Tailwind CSS v4 config
│   ├── docker-compose.yml           # Docker services (optional)
│   └── tsconfig.json                # TypeScript configuration
└── [Scripts]
    ├── start_hybrid.sh              # Start hybrid mode (vector + AI)
    ├── start_trained.sh             # Start trained model mode
    ├── start_training.sh            # Train your own model
    ├── prepare_training_data.sh     # Prepare bio data for training
    ├── start_all_servers.sh         # Legacy: Start all services
    └── start_api_server.sh          # Start API server only
```

## Available Scripts

### Development Scripts
- `npm run dev` - Start development server with Turbopack
- `npm run build` - Build the application for production
- `npm run start` - Start the production server
- `npm run lint` - Run ESLint for code quality

### Server Management Scripts
- `./start_all_servers.sh` - Start all backend services (API + Trained Model)
- `./start_api_server.sh` - Start only the main API server
- `cd python && ./setup.sh` - Initial Python environment setup
- `cd python && ./activate.sh` - Activate Python virtual environment

## Server Management

### Using Shell Scripts

The project includes convenient shell scripts for managing backend services:

#### `start_all_servers.sh`
- Starts both API server (port 8001) and trained model server (port 8002)
- Checks if Ollama is running and provides guidance if not
- Prevents duplicate server instances
- Creates log files for debugging (`python/api_server.log`)
- Shows status of all services with helpful URLs

#### `start_api_server.sh`
- Starts only the main API server on port 8001
- Provides clear instructions for required services
- Shows API documentation URL

### Managing Services

#### Check Service Status
```bash
# Check if services are running
lsof -i :8001  # API Server
lsof -i :11434 # Ollama
lsof -i :8003  # MLX Model Server

# View logs
tail -f python/api_server.log
```

#### Stop Services
```bash
# Stop individual services
lsof -ti:8001 | xargs kill  # API Server
lsof -ti:8003 | xargs kill  # MLX Model Server

# Stop all Python API processes
pkill -f "python.*api_server"
pkill -f "python.*mlx_model_server"
```

## Configuration

### Ollama Configuration

The application expects Ollama to be running on `http://127.0.0.1:11434` by default. You can modify this in your `.env.local` file:

```env
OLLAMA_PATH_API=http://your-ollama-host:port/api
```

### Python API Configuration

The FastAPI server runs on port 8001 by default. Key configuration options:

```python
# In python/api/api_server.py
OLLAMA_API_URL = "http://localhost:11434/api"
CHROMA_PERSIST_DIR = "../chroma_db"
MAX_SUGGESTIONS = 3
MIN_SUGGESTION_LENGTH = 8  # Minimum words per suggestion
```

### Vector Database Configuration

ChromaDB is configured to persist data locally:
- **Storage location**: `python/chroma_db/`
- **Collection name**: `bio_embeddings`
- **Embedding function**: Default (all-MiniLM-L6-v2)
- **Bio count**: ~5000 entries from `data/bio.json`

### Advanced Configuration

**Spell Check Settings:**
- 800ms debounce delay with suggestion caching
- Custom dictionary with localStorage persistence
- Word mapping for learning corrections
- 80+ automatic contraction suggestions (dont → don't, youre → you're, etc.)

**AI Model Settings:**
- Primary: `gemma3:12b` via Ollama
- Vector DB: ChromaDB with ~5000 bios
- Adaptive debouncing: 50-400ms
- Smart caching: 5-minute TTL


## Troubleshooting

### Common Issues

1. **Ollama not responding**:

   - Ensure Ollama service is running: `ollama serve`
   - Check if the model is pulled: `ollama list`
   - Verify the API endpoint in `.env.local`

2. **Model not found**:

   - Pull the model: `ollama pull gemma3:12b`
   - Restart Ollama service

3. **Slow responses**:
   - This is normal for local AI models
   - Consider using a more powerful machine or adjusting the debounce delay

4. **Spell check not working**:
   - Ensure dictionary files are present in `public/dictionaries/en_US/`
   - Check browser console for any dictionary loading errors
   - Try refreshing the page to reinitialize the spell checker
   - Clear localStorage if custom dictionary is corrupted: `localStorage.clear()`

5. **Performance issues while typing**:
   - The app uses sophisticated debouncing to prevent lag
   - Text Feature Coordinator manages feature conflicts automatically
   - If typing feels slow, check if other applications are using high CPU
   - Progressive debouncing adapts timing based on text length

6. **Features conflicting with each other**:
   - The Text Feature Coordinator should prevent this automatically
   - If issues persist, try refreshing the page to reset coordinator state
   - Check browser console for any coordination errors

## Dependencies

### Frontend Dependencies

- **Next.js 15.3.3** - React framework with App Router and Server Actions
- **React 19** - Latest UI library with advanced hooks
- **React Hook Form 7.58.0** - Sophisticated form management
- **typo-js 1.2.5** - Spell checking with Hunspell dictionary support
- **use-debounce 10.0.5** - Performance optimization utility

### Python Backend Dependencies

- **FastAPI** - High-performance web framework for building APIs
- **ChromaDB 0.4.24** - Vector database for storing bio embeddings
- **Uvicorn** - ASGI server for running FastAPI
- **Transformers** - Hugging Face library for model training
- **PyTorch** - Deep learning framework for model fine-tuning
- **Sentence Transformers** - For generating embeddings

### Development Dependencies

- **TypeScript 5** - Comprehensive type safety
- **Tailwind CSS 4** - Latest version with PostCSS
- **ESLint 9** - Advanced code linting with Next.js config

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Model Training (Optional)

The project includes comprehensive model training capabilities for custom bio generation:

## Additional Documentation

### Core Documentation
- **[how_to_use.md](app_docs/how_to_use.md)** - Comprehensive guide for using and training the system
- **[API_SERVER_GUIDE.md](app_docs/API_SERVER_GUIDE.md)** - Detailed API documentation
- **[progress-overview.md](app_docs/progress-overview.md)** - Development progress and architecture details

### Feature Documentation
- **[kick-upgrade.md](kick-upgrade.md)** - Complete kick detection v2 implementation (Phase 1-2)
- **[kick.md](app_docs/kick.md)** - Original kick detection strategy
- **[kick-implementation-summary.md](app_docs/kick-implementation-summary.md)** - Quick overview of kick detection
- **[AUTOCOMPLETE_OPTIMIZATIONS.md](app_docs/AUTOCOMPLETE_OPTIMIZATIONS.md)** - Performance optimization details
- **[simon_updates.md](app_docs/simon_updates.md)** - Detailed changelog of all updates

## Acknowledgments

- [Ollama](https://ollama.ai) for providing local AI model hosting
- [Google Gemma](https://ai.google.dev/gemma) for the powerful language model
- [ChromaDB](https://www.trychroma.com/) for the efficient vector database
- [FastAPI](https://fastapi.tiangolo.com/) for the high-performance API framework
- [typo-js](https://github.com/cfinke/Typo.js) for excellent spell checking capabilities
- [Hunspell](http://hunspell.github.io/) for comprehensive dictionary support
- [Next.js](https://nextjs.org) team for the excellent framework
