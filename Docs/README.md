# AI Bio Autocomplete Documentation

Welcome to the AI Bio Autocomplete documentation! This guide will help you understand, set up, and work with our AI-powered bio completion system.

## 📚 Documentation Overview

### Getting Started
- **[Quick Start Guide](getting_started.md)** - Get up and running in 5 minutes
- **[API Server Guide](API_SERVER_GUIDE.md)** - Python backend setup and endpoints

### Technical Guides
- **[Architecture Guide](architecture-guide.md)** - System design and patterns
- **[Training Guide](training-guide.md)** - Train your own models
- **[Kick Detection Guide](kick-detection-guide.md)** - Content filtering system
- **[How to Use](how_to_use.md)** - Comprehensive usage examples

### Performance & Optimization
- **[Autocomplete Optimizations](AUTOCOMPLETE_OPTIMIZATIONS.md)** - Performance tuning strategies
- **[Future Enhancements](FUTURE_ENHANCEMENT_trained_model_fix.md)** - Planned improvements

### Project Information
- **[Project Instructions](../CLAUDE.md)** - Claude.ai integration guide
- **[Archived Docs](archive/)** - Historical documentation

## 🚀 Quick Links

| Task | Command/Link |
|------|-------------|
| Start all services | `./start_all_servers.sh` |
| Start dev server | `npm run dev` |
| View app | http://localhost:3000 |
| API docs | http://localhost:8001/docs |
| Python setup | `cd python && ./setup.sh` |

## 🏗️ System Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Next.js App    │────▶│  Python API      │────▶│  Ollama LLM     │
│  (Port 3000)    │     │  (Port 8001)     │     │  (Port 11434)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │  ChromaDB        │
                        │  Vector Search   │
                        └──────────────────┘
```

## 📖 For New Developers

1. **Start Here**: Read the [Quick Start Guide](getting_started.md)
2. **Understand the System**: Review [How to Use](how_to_use.md)
3. **Dive Deeper**: Check the [Architecture Overview](../CLAUDE.md)
4. **Run the App**: Follow the [API Server Guide](API_SERVER_GUIDE.md)

## 🔧 Common Tasks

### Running the Application
```bash
# Start all backend services
./start_all_servers.sh

# In another terminal, start the frontend
npm run dev
```

### Training Your Own Model
See [Training Models Guide](training_llm_local.md) for detailed instructions on fine-tuning models with your data.

### Understanding Features
- **AI Autocomplete**: Hybrid vector search + LLM generation
- **Spell Checking**: Advanced typo detection with custom dictionary
- **Kick Detection**: Sophisticated content filtering
- **Performance**: Sub-150ms response times

## 📝 Documentation Standards

All documentation follows these principles:
- **Clarity**: Simple language, clear examples
- **Completeness**: All necessary information included
- **Currency**: Regularly updated with latest changes
- **Organization**: Logical structure and flow

## 🤝 Contributing

When updating documentation:
1. Keep language simple and direct
2. Include practical examples
3. Update related docs when making changes
4. Test all commands and code snippets

---

*Last updated: July 2025*