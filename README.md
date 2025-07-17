# AI Bio Autocomplete with Hybrid Vector Search

A sophisticated AI-powered bio autocomplete system built with Next.js 15, React 19, and Python FastAPI. Combines vector database search (ChromaDB) with LLM generation (Ollama Gemma 3 12B) for high-quality, contextually relevant bio completions. Features intelligent autocomplete, advanced spell checking, kick.com link detection, and a hybrid approach optimized for swinger community bios.

## 🚀 Recent Updates

### Latest Features (2025)
- **🛡️ Kick.com Link Detection v2** - Enhanced detection with 40+ patterns including phonetic variations and zero-width character support
- **🔧 Enhanced ChromaDB** - Now indexes all 5000 bios for better vector search results
- **✨ Phonetic Pattern Detection** - Catches sound-alike variations (keek, kyck, keak)
- **🐛 Build Fixes** - Zero errors in production build with full TypeScript compliance

## Features

### 🛡️ Kick.com Link Detection System v2
- **Advanced URL detection** with 40+ obfuscation patterns (up from 28)
- **Phonetic detection** - Catches sound-alike variations (keek, kyck, keak, kik)
- **Zero-width character detection** - Identifies invisible Unicode obfuscation
- **Multi-layer detection** - Pattern matching, fuzzy matching, homoglyph, and phonetic
- **Real-time warnings** with confidence levels (Low/Medium/High)
- **Smart pattern recognition** - Spaces, dots, parentheses, special chars, leetspeak
- **False positive prevention** - Excludes legitimate words (kayak, peek, etc.)
- **Performance optimized** - Sub-5ms detection with intelligent caching

### 🤖 Hybrid AI Autocomplete System
- **Dual approach** - Combines ChromaDB vector search with Ollama LLM generation
- **Fast response times** - 100-150ms hybrid performance (vs 200-500ms LLM-only)
- **Context-aware suggestions** using 5000 bio examples in vector database
- **Smart triggering** - activates after 3-5 words with adaptive debouncing
- **Inline suggestion display** with layered textarea approach
- **Tab key acceptance** with intelligent spacing
- **Quality filtering** - Minimum 8-word suggestions with complete thoughts
- **Streaming responses** - Real-time character-by-character display (60-80% faster perceived latency)

### ✍️ Professional Spell Checking
- **typo-js integration** with English Hunspell dictionaries
- **Custom dictionary system** with localStorage persistence
- **Word mapping functionality** - learns user corrections
- **Contraction handling** - automatically suggests 80+ common contractions
- **Click-to-correct interface** with intelligent popup positioning
- **Performance optimized** with 800ms debouncing and suggestion caching
- **Auto-correction dictionary** - 80+ common misspellings with preserved capitalization

- **Pattern learning** - Logs detection attempts for continuous improvement

### 🎯 Intelligent Feature Coordination
- **Text Feature Coordinator** prevents conflicts between autocomplete and spellcheck
- **Adaptive timing system** with feature-specific lock durations
- **Seamless multi-feature operation** - autocomplete and spellcheck work together
- **Memory management** with proper cleanup and state handling
- **Auto-capitalization** - Smart sentence and pronoun capitalization

### 🚀 Modern Architecture
- **Next.js 15** with App Router and Server Actions
- **React 19** with advanced hooks and TypeScript
- **Tailwind CSS v4** with responsive design
- **Mobile-optimized UX** with 16px fonts to prevent zoom
- **Dynamic textarea resizing** with smooth 300ms transitions
- **Performance optimized** with progressive debouncing and memoization

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

1. **Clone the repository**:

   ```bash
   git clone <your-repository-url>
   cd ai-train-llm
   ```

2. **Install Node.js dependencies**:

   ```bash
   npm install
   # or
   yarn install
   ```

3. **Set up Python environment**:

   ```bash
   cd python
   python3 -m venv venv
   source venv/bin/activate  # On Mac/Linux
   # or
   venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   cd ..
   ```

4. **Set up environment variables**:
   Create a `.env.local` file in the root directory:
   ```env
   OLLAMA_PATH_API=http://127.0.0.1:11434/api
   # Optional: Enable fine-tuned model integration
   NEXT_PUBLIC_USE_FINETUNED_MODEL=true
   # Optional: OpenAI API key for fallback
   NEXT_PUBLIC_OPENAI_API_KEY=your-key-here
   ```

5. **Initialize the vector database**:

   ```bash
   cd python/vector_db
   python setup_chromadb.py
   cd ../..
   ```

6. **Set up spell check dictionaries**:
   The application includes English dictionaries in the `public/dictionaries/en_US/` folder:
   - `en_US.aff` - Affix rules file
   - `en_US.dic` - Dictionary words file

## Running the Full Stack

### Quick Start (All Services)
```bash
# Start all backend services with one command
./start_all_servers.sh

# Then in another terminal, start Next.js
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

#### 3. **Start Trained Model Server** (Terminal 3 - Optional):
```bash
cd python
python -m uvicorn api.trained_model_server:app --port 8002
```
The trained model server will run on `http://localhost:8002`

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

1. **Open your browser** and navigate to `http://localhost:3000`

2. **Start typing** in the bio description field to experience:
   - **AI Autocomplete**: After 3-5 complete words, AI suggestions appear as gray inline text
   - **Spell Check**: Misspelled words show red dotted underlines with click-to-correct
   - **Contraction Help**: Type "dont" and see automatic "don't" suggestions
   - **Custom Dictionary**: Add frequently used words to your personal dictionary
   - **Auto-capitalization**: Smart sentence formatting applied in real-time
   - **Kick.com Detection**: Real-time warnings for prohibited link variations

3. **Advanced interactions**:
   - Press **Tab** to accept AI suggestions with proper spacing
   - **Click** misspelled words for instant popup with corrections
   - **Right-click** words to add them to your custom dictionary
   - **Type naturally** - the Text Feature Coordinator prevents interference
   - **Mobile-friendly** - all features work seamlessly on touch devices

## Python API Server

The application includes a FastAPI server that provides the hybrid autocomplete functionality:

### API Endpoints

#### Main API Server (Port 8001)
- **GET /** - Health check endpoint
- **POST /api/autocomplete** - Vector-only autocomplete suggestions
- **POST /api/autocomplete/hybrid** - Hybrid autocomplete (vector + LLM)
- **GET /api/stats** - Database statistics

#### Trained Model Server (Port 8002 - Optional)
- **GET /health** - Health check endpoint
- **POST /api/autocomplete/trained** - Autocomplete using fine-tuned GPT-2 models

### Hybrid Approach

The hybrid autocomplete system combines:
1. **Vector Search** - Fast exact matches from ~5000 bio examples using ChromaDB
2. **LLM Generation** - Creative completions using Ollama Gemma 3 12B
3. **Quality Filtering** - Ensures suggestions are complete thoughts (8+ words)

### Performance Metrics

- **Standard Mode** (`/`): 100-150ms response time
- **Vector Search**: ~100ms for similarity matching
- **LLM Generation**: 200-500ms (without optimization)
- **Fine-tuned Model**: 80-120ms (faster than base LLM)
- **Cache Hit Rate**: 90% reduction in API calls with smart caching
- **Adaptive Debouncing**: 50-400ms based on typing speed
- **Streaming Latency**: Character-by-character display for perceived speed

### API Documentation

- **Main API Server**: `http://localhost:8001/docs`
- **Trained Model Server**: `http://localhost:8002/docs` (when running)

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

### Sophisticated Hook Architecture
The application uses a **5-hook system** for optimal performance and feature coordination:

1. **`useFormAutocomplete`** - Main form logic with AI integration
2. **`useSpellCheck`** - Core spell checking with typo-js and custom dictionaries
3. **`useDebouncedSpellCheck`** - Performance-optimized wrapper with caching
4. **`useTextFeatureCoordinator`** - Prevents conflicts between features
5. **`useKickDetection`** - Real-time pattern matching for prohibited links

### Text Feature Coordination System
The **Text Feature Coordinator** manages three text features:
- **AUTOCOMPLETE** - AI-powered text suggestions
- **SPELLCHECK** - Real-time spell checking
- **CAPITALIZATION** - Smart sentence formatting

**Key Coordination Features:**
- **Feature locking** with adaptive durations (200ms for autocomplete, configurable for others)
- **Coexistence support** - autocomplete and spellcheck can run simultaneously
- **Memory management** - automatic cleanup of timeouts and state
- **Conflict prevention** - ensures features don't interfere with each other

### AI Autocomplete System
- **Layered textarea approach** for inline suggestion display
- **Word completion detection** - waits for complete words before suggesting
- **Progressive debouncing** - 50-400ms adaptive delay based on typing speed
- **Vector database integration** with ChromaDB for fast similarity search
- **Server Actions** communicate with Ollama Gemma 3 12B model
- **Intelligent spacing** - proper handling of tab acceptance and word boundaries
- **Streaming responses** - Character-by-character display in optimized mode
- **Smart caching** - 90% reduction in redundant API calls with 5-minute TTL

### Kick.com Link Detection System v2
- **Pattern matching engine** - Detects 40+ real-world obfuscation patterns:
  - Character spacing: `k i k`, `k.i.k`, `k-i-k`, `k....i....k`
  - Character substitution: `k1k`, `k!k`, `klk` (l for i)
  - Character insertion: `kiik`, `kiiik`, `killk`
  - Phonetic variations: `keek`, `kyck`, `keak`, `kik`
  - Special formatting: `k(i)k`, `k(__ei__)k`, `k(._i_.)k`
  - Extended patterns: `k..ee..k`, `k._.-i-._.k`
  - Zero-width characters: `k​i​c​k` (with invisible Unicode)
- **Multi-layer detection approach**:
  - Pattern matching with 40+ regex patterns
  - Fuzzy matching with Levenshtein distance
  - Homoglyph detection for Unicode look-alikes
  - Phonetic matching for sound-alike variations
  - Zero-width character normalization
- **False positive prevention** - Excludes legitimate words (kayak, peek, etc.)
- **Context analysis** - Increases confidence with streaming-related keywords
- **Performance optimized** - Sub-5ms detection with progressive checking
- **Non-intrusive warnings** - Color-coded by confidence (red/orange/yellow)
- **Position tracking** - Accurate even with invisible characters
- **Learning system** - Logs patterns for continuous improvement

### Advanced Spell Check System
- **typo-js integration** with English Hunspell dictionaries (`en_US.aff`, `en_US.dic`)
- **Custom dictionary service** with localStorage persistence
  - Add/remove custom words
  - Word mapping system for learning corrections
  - Persistent storage across browser sessions
- **Contraction handling** - automatic suggestions for 80+ common contractions:
  - `dont` → `don't`, `do not`
  - `youre` → `you're`, `you are`
  - `wont` → `won't`, `will not`
  - And many more...
- **Click-to-correct interface** with intelligent popup positioning
- **Performance optimizations:**
  - 800ms debounce delay
  - Suggestion caching to prevent repeated lookups
  - Memoization to prevent unnecessary re-renders
  - Progressive debouncing based on text length

### Smart Auto-Capitalization
- **Sentence boundary detection** with proper punctuation handling
- **Real-time processing** without disrupting typing flow
- **Context-aware rules:**
  - First letter of sentences
  - After periods, exclamation marks, question marks
  - Pronoun "I" capitalization
- **Seamless integration** with autocomplete and spell check features

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
│   │   ├── api_server.py            # FastAPI hybrid autocomplete server
│   │   └── trained_model_server.py  # FastAPI server for fine-tuned models
│   ├── vector_db/
│   │   ├── setup_chromadb.py        # Initialize vector database
│   │   └── vector_search.py         # Vector search implementation
│   ├── mlx_training/                # Model training scripts
│   │   ├── train_bio_improved.py    # GPT-2 fine-tuning with LoRA
│   │   ├── train_distilgpt2.py      # DistilGPT2 fine-tuning
│   │   ├── bio_gpt2_improved/       # Fine-tuned GPT-2 model
│   │   ├── bio_distilgpt2_finetuned/# Fine-tuned DistilGPT2 model
│   │   ├── bio_gpt2_finetuned/      # Standard GPT-2 fine-tuned
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
    ├── start_all_servers.sh         # Start all backend services
    └── start_api_server.sh          # Start main API server only
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
- Creates log files for debugging (`python/api_server.log`, `python/trained_model_server.log`)
- Shows status of all services with helpful URLs

#### `start_api_server.sh`
- Starts only the main API server on port 8001
- Provides clear instructions for required services
- Useful when you don't need the trained model server
- Shows API documentation URL

### Managing Services

#### Check Service Status
```bash
# Check if services are running
lsof -i :8001  # API Server
lsof -i :8002  # Trained Model Server
lsof -i :11434 # Ollama

# View logs
tail -f python/api_server.log
tail -f python/trained_model_server.log
```

#### Stop Services
```bash
# Stop individual services
lsof -ti:8001 | xargs kill  # API Server
lsof -ti:8002 | xargs kill  # Trained Model Server

# Stop all Python API processes
pkill -f "python.*api_server"
pkill -f "uvicorn.*trained_model"
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

### Fine-tuned Model Configuration

The trained model server provides access to fine-tuned GPT-2 models:

```python
# In python/api/trained_model_server.py
model_path = "mlx_training/bio_distilgpt2_finetuned"  # Default model
device = "mps"  # macOS Metal Performance Shaders (or "cpu")
```

**Available Models:**
- `bio_distilgpt2_finetuned` - Lightweight, fast inference (default)
- `bio_gpt2_improved` - Larger model with LoRA fine-tuning
- `bio_gpt2_finetuned` - Standard GPT-2 fine-tuning

To enable fine-tuned model integration in the frontend:
```env
NEXT_PUBLIC_USE_FINETUNED_MODEL=true
```

### Vector Database Configuration

ChromaDB is configured to persist data locally:
- **Storage location**: `python/chroma_db/`
- **Collection name**: `bio_embeddings`
- **Embedding function**: Default (all-MiniLM-L6-v2)
- **Bio count**: ~5000 entries from `data/bio.json`

### Spell Check Configuration

Spell checking features advanced configuration options:

**Core Settings:**
- **Debounce delay**: 800ms for optimal performance
- **Dictionary caching**: Suggestions cached to prevent repeated lookups
- **Custom dictionary**: Persistent localStorage-based word storage
- **Word mapping**: Learn and remember user corrections

**Contraction Handling:**
The system automatically handles 80+ common contractions:
```javascript
// Examples of automatic contraction suggestions
dont     → don't, do not
youre    → you're, you are
wont     → won't, will not
havent   → haven't, have not
its      → it's, it is, it has
hes      → he's, he is, he has
shes     → she's, she is, she has
mustnt   → mustn't, must not
```

**Custom Dictionary Features:**
- **Add words**: Right-click misspelled words to add to dictionary
- **Word mappings**: System learns your preferred corrections
- **Persistent storage**: Dictionary survives browser restarts
- **Import/Export**: Backup and restore custom words

### AI Model Configuration

**Default Setup:**
- **Primary Model**: `gemma3:12b` via Ollama (local inference)
- **Vector Database**: ChromaDB with ~5000 bio embeddings
- **OpenAI Integration**: Available as fallback option

**Configuration Options:**
```javascript
// In actions/ai-text.ts - modify model settings
const model = "gemma3:12b";  // Change model here
const maxTokens = 50;        // Adjust response length
const temperature = 0.7;     // Control creativity
```

**Performance Tuning:**
- **Progressive debouncing**: Adapts timing based on text length (50-400ms)
- **Context window**: Optimized for bio completion tasks
- **Feature coordination**: Automatic AI pause during spell check operations
- **Smart caching**: 5-minute TTL cache for repeated prompts

## Recent Updates (2025)

### 🛡️ Kick.com Link Detection v2 (Phase 1-2 Complete)
- Expanded pattern matching from 28 to 40+ obfuscation techniques
- Added phonetic detection for sound-alike variations (keek, kyck, keak)
- Implemented zero-width character detection for invisible Unicode obfuscation
- Enhanced parentheses and extended character gap patterns
- Fixed critical bypasses and improved false positive prevention
- Maintained sub-5ms performance with intelligent caching

### ⚡ Performance Optimizations
- **Streaming Responses**: 60-80% faster perceived latency with character-by-character display
- **Adaptive Debouncing**: Dynamic 50-400ms delays based on typing speed
- **Smart Caching**: 90% reduction in redundant API calls
- **React 19 Features**: Non-blocking updates with useTransition and useDeferredValue

### ✨ Enhanced User Experience
- Reduced autocomplete trigger from 5 to 3-4 words
- Improved spell correction with 80+ common misspellings
- Auto-capitalization for sentences and pronouns
- Prevention of word repetition in suggestions
- Mobile-optimized 16px fonts to prevent zoom
- Swinger-specific language patterns for authentic suggestions

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

## Key Features In Detail

### Smart Spell Checking
- **Professional-grade accuracy** using typo-js with Hunspell dictionaries
- **Click-to-correct interface** - no right-click needed, just click misspelled words
- **Custom dictionary system** with persistent localStorage storage
- **Word mapping functionality** - learns and remembers your corrections
- **35+ contraction suggestions** - automatic handling of missing apostrophes
- **Performance optimized** with 800ms debouncing and intelligent caching
- **Non-intrusive visual indicators** - dotted red underlines without typing disruption

### AI-Powered Autocomplete  
- **Context-aware suggestions** using Gemma 3 12B model for relevance
- **Layered textarea approach** for seamless inline suggestion display
- **Smart triggering system** - activates after 5+ complete words
- **Vector database integration** with ChromaDB for enhanced understanding
- **Intelligent spacing logic** - proper tab acceptance and word boundaries
- **Progressive debouncing** - 500ms delay with adaptive timing

### Text Feature Coordination
- **Conflict prevention system** - prevents autocomplete and spellcheck interference
- **Adaptive feature locking** with customizable durations per feature type
- **Memory management** - automatic cleanup of timeouts and state
- **Coexistence support** - allows multiple features to work simultaneously
- **Performance monitoring** - tracks and optimizes feature interactions

### Mobile-Optimized Experience
- **16px font sizes** prevent unwanted mobile browser zoom
- **Dynamic textarea resizing** with smooth 300ms transitions
- **Touch-friendly interface** with properly sized click targets
- **Responsive design** adapts to all screen sizes
- **Optimized debouncing** for mobile keyboard behavior
- **Battery-conscious processing** with intelligent feature management

## Model Training (Optional)

The project includes comprehensive model training capabilities for custom bio generation:

### Available Pre-trained Models
- **bio_gpt2_improved** - Fine-tuned GPT-2 on bio data
- **bio_distilgpt2_finetuned** - Lighter DistilGPT2 variant
- **bio_gpt2_finetuned** - Standard GPT-2 fine-tuning

### Training Your Own Model

1. **Prepare training data**:
   ```bash
   cd python/mlx_training
   python prepare_mlx_data.py
   ```

2. **Train the model**:
   ```bash
   # GPT-2 fine-tuning with LoRA
   python train_bio_improved.py
   
   # DistilGPT2 fine-tuning (faster, lighter)
   python train_distilgpt2.py
   
   # Quick testing with smaller dataset
   python train_simple.py
   ```

3. **Deploy the trained model**:
   - Models are automatically saved in respective directories
   - Start the trained model server on port 8002
   - Enable via `NEXT_PUBLIC_USE_FINETUNED_MODEL=true`
   - Access at `/api/autocomplete/trained` endpoint

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
