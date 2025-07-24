# AI Bio Autocomplete Documentation Hub

Comprehensive documentation for the AI-powered bio text completion system featuring hybrid vector search and fine-tuned language models.

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd ai-train-llm
npm install

# Choose your mode and start
./start_hybrid.sh    # Best quality (ChromaDB + Ollama)
./start_trained.sh   # Fastest (Fine-tuned Llama 3.2)

# Run the app
npm run dev
```

Visit http://localhost:3000 and start typing!

## 📚 Documentation Library

### 🎯 Getting Started
- **[Getting Started Guide](./GETTING_STARTED.md)** - Complete setup and installation
- **[Quick Start Guide](./GETTING_STARTED.md#quick-start)** - Running in 5 minutes
- **[FAQ](./FAQ.md)** - Frequently asked questions

### 🏃 Running & Operations
- **[Running the App](./RUNNING_THE_APP.md)** - Hybrid vs Trained modes explained
- **[Configuration Guide](./CONFIGURATION.md)** - All settings and options
- **[Troubleshooting](./TROUBLESHOOTING.md)** - Fix common problems

### 🔧 Development
- **[Development Guide](./DEVELOPMENT_GUIDE.md)** - Code standards and workflows
- **[API Reference](./API_REFERENCE.md)** - Backend endpoints and integration
- **[Architecture Overview](./ARCHITECTURE.md)** - System design and data flow

### 🤖 AI & Training
- **[Training Guide](./TRAINING_GUIDE.md)** - Train custom models
- **[ChromaDB Guide](./CHROMADB_GUIDE.md)** - Vector database operations

## 🎯 Quick Navigation

| I want to... | Start here |
|-------------|------------|
| **First time setup** | [Getting Started](./GETTING_STARTED.md) |
| **Understand the system** | [Architecture](./ARCHITECTURE.md) |
| **Fix autocomplete issues** | [Troubleshooting](./TROUBLESHOOTING.md) |
| **Train a custom model** | [Training Guide](./TRAINING_GUIDE.md) |
| **Configure settings** | [Configuration](./CONFIGURATION.md) |
| **Contribute code** | [Development Guide](./DEVELOPMENT_GUIDE.md) |
| **Work with the database** | [ChromaDB Guide](./CHROMADB_GUIDE.md) |
| **Common questions** | [FAQ](./FAQ.md) |

## 🏗 System Overview

### Architecture
- **Frontend**: Next.js 15 + React 19 with TypeScript
- **Backend**: FastAPI (Python) with dual server architecture
- **AI Models**: Ollama (Gemma 3) + Fine-tuned Llama 3.2
- **Database**: ChromaDB for vector search
- **Features**: Autocomplete, spell check, content filtering

### Key Features
✅ **8-20 word bio completions** - Natural sentence completion  
✅ **100-150ms response time** - Fast and responsive  
✅ **Hybrid approach** - Vector search + AI generation  
✅ **Fine-tuned models** - Custom trained on bio data  
✅ **Advanced UI** - Spell check, content filtering, mobile optimized  

## 💻 Development Status

### Working Features
- Hybrid autocomplete (ChromaDB + Ollama)
- Fine-tuned model inference (MLX)
- Spell checking with contractions
- Content filtering (40+ patterns)
- 5-minute response caching
- Mobile-responsive design

### Known Issues
- LLM sometimes exceeds 20-word limit
- Requires manual Ollama installation
- MLX models need Apple Silicon

## 🛠 Essential Commands

```bash
# Development
npm run dev              # Start Next.js dev server
npm run build           # Production build
npm run lint            # Check code quality

# Backend Services
./start_hybrid.sh       # Vector search + AI mode
./start_trained.sh      # Fine-tuned model mode

# Training
./prepare_hq_data_fast.sh        # Prepare quality data
./start_training_mlx_community.sh # Train new model

# Maintenance
cd python/vector_db && python setup_chromadb.py  # Rebuild database
```

## 📊 Performance Metrics

| Mode | Response Time | Quality | Best For |
|------|--------------|---------|----------|
| **Hybrid** | 100-150ms | High | General use |
| **Trained (3B)** | 100-150ms | Good | Consistent style |
| **Trained (1B)** | 50-100ms | Good | Speed priority |

## 🔍 Documentation Standards

Our documentation follows these principles:
- **Practical** - Real examples and commands
- **Comprehensive** - Covers all features
- **Up-to-date** - Reflects current code
- **Accessible** - Clear for all skill levels

## 📝 Contributing

1. Read [Development Guide](./DEVELOPMENT_GUIDE.md)
2. Check existing issues
3. Follow code standards
4. Test thoroughly
5. Update relevant docs

## 🆘 Getting Help

1. **Check [FAQ](./FAQ.md)** - Common questions answered
2. **Read [Troubleshooting](./TROUBLESHOOTING.md)** - Known issues and fixes
3. **Search docs** - Use Ctrl+F to find keywords
4. **Check logs** - `/tmp/api_server.log` and browser console
5. **Open issue** - Include error messages and steps to reproduce

## 📅 Last Updated

December 2024 - Documentation current as of v1.0.0

---

*Built with ❤️ for fast, intelligent bio completion*