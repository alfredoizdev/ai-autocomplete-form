# Frequently Asked Questions (FAQ)

Common questions and answers about the AI Bio Autocomplete system.

## 🚀 Getting Started

### Q: What's the quickest way to get the app running?
**A:** Use the quick start scripts:
```bash
# For best quality (vector search + AI)
./start_hybrid.sh

# For fastest speed (fine-tuned model)
./start_trained.sh

# Then in a new terminal
npm run dev
```
Visit http://localhost:3000 and start typing!

### Q: What's the difference between hybrid and trained mode?
**A:** 
- **Hybrid Mode**: Searches existing bios for context, then generates new text. Better quality, ~150ms response.
- **Trained Mode**: Uses a fine-tuned model directly. Faster (~100ms), more consistent style.

### Q: Do I need special hardware?
**A:** 
- Basic usage: Any modern computer (8GB RAM minimum)
- Training models: Apple Silicon Mac (M1/M2/M3) or NVIDIA GPU
- Ollama: 16GB RAM recommended for gemma3:12b

## 🔧 Troubleshooting

### Q: Why is autocomplete not working?
**A:** Check these common issues:
1. Ensure backend is running (`ps aux | grep python`)
2. Type at least 3-4 words to trigger autocomplete
3. Check browser console for errors
4. Verify Ollama is running: `curl http://localhost:11434/api/tags`

### Q: Why are suggestions too long (more than 20 words)?
**A:** This is a known LLM behavior. The app enforces limits by:
1. Setting max_tokens=25 in API calls
2. Truncating responses to 20 words
3. Using prompts that emphasize "8-20 words ONLY"

Try restarting the API server if issues persist.

### Q: How do I fix "Address already in use" error?
**A:** Kill the existing process:
```bash
# Find the process
lsof -i :8001  # or :8003 for MLX server

# Kill it
kill -9 <PID>

# Or use the one-liner
lsof -ti:8001 | xargs kill -9
```

## 💡 Features

### Q: How does spell checking work?
**A:** 
- Uses Hunspell dictionaries (US English)
- Supports 80+ contractions (don't, won't, etc.)
- Custom dictionary for user-added words
- 800ms debounce to avoid interference with typing

### Q: What is kick detection?
**A:** A content filter that blocks certain website references (kick.com) using 40+ patterns to detect obfuscation attempts. It runs in <5ms and shows warnings to users.

### Q: Can I disable spell check or kick detection?
**A:** Yes, through environment variables:
```bash
# In .env.local
NEXT_PUBLIC_ENABLE_SPELL_CHECK=false
NEXT_PUBLIC_ENABLE_KICK_DETECTION=false
```

## 🤖 AI & Models

### Q: Which AI model should I use?
**A:** 
- **For quality**: Gemma 3 12B (via Ollama in hybrid mode)
- **For speed**: Llama 3.2 3B fine-tuned (trained mode)
- **For efficiency**: Llama 3.2 1B fine-tuned

### Q: Can I use OpenAI instead of Ollama?
**A:** Yes! Add to `.env.local`:
```bash
NEXT_PUBLIC_OPENAI_API_KEY=your-key-here
```
The app will fall back to OpenAI if Ollama fails.

### Q: How do I train my own model?
**A:** Quick training process:
```bash
# 1. Prepare high-quality data
./prepare_hq_data_fast.sh

# 2. Start training (takes 2-4 hours)
./start_training_mlx_community.sh
```

## 📊 Data & Privacy

### Q: What happens to the bio data I type?
**A:** 
- Nothing is stored permanently
- Temporary cache (5 minutes) for performance
- No external API calls except to Ollama/OpenAI
- All processing happens locally

### Q: Can I add my own training data?
**A:** Yes! Add your data to `/data/` as a CSV file, then:
1. Update the data preparation script to point to your file
2. Run `./prepare_hq_data_fast.sh`
3. Train a new model

### Q: How much training data do I need?
**A:** 
- Minimum: 1,000 examples
- Recommended: 5,000+ examples
- Current dataset: 4,500 high-quality examples

## ⚡ Performance

### Q: How can I make autocomplete faster?
**A:** 
1. Use trained mode instead of hybrid
2. Reduce debounce time in settings
3. Use the 1B model instead of 3B
4. Ensure nothing else is using ports 8001/8003

### Q: What are typical response times?
**A:** 
- Trained mode (1B): 50-100ms
- Trained mode (3B): 100-150ms  
- Hybrid mode: 100-200ms
- First request: +200ms (model loading)

### Q: How does caching work?
**A:** 
- 5-minute TTL for identical prompts
- Dramatically reduces API calls (90% cache hit rate)
- Clears on server restart
- No persistent storage

## 🛠 Development

### Q: How do I add a new feature?
**A:** 
1. Check `DEVELOPMENT_GUIDE.md` for code standards
2. Add hooks to `/hooks/` for React features
3. Update `useTextFeatureCoordinator` for UI features
4. Add endpoints to `api_server.py` for backend features

### Q: Can I use this with other frameworks?
**A:** Yes! The API is framework-agnostic:
```bash
# Any framework can call the API
curl -X POST http://localhost:8001/api/autocomplete/hybrid \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Looking for"}'
```

### Q: How do I run tests?
**A:** Currently no automated tests. Recommended testing:
```bash
# Lint check
npm run lint

# Type check  
npm run build

# Manual testing
# - Check autocomplete speed (<150ms)
# - Verify 8-20 word responses
# - Test spell check on typos
```

## 🌐 Deployment

### Q: Can I deploy this to the cloud?
**A:** Yes, but note:
- MLX requires Apple Silicon (use AWS Mac instances)
- Ollama needs significant RAM (16GB+)
- Consider using trained mode only for easier deployment

### Q: How do I deploy with Docker?
**A:** Full Docker support coming soon. Current approach:
```bash
# Backend services can be containerized
cd python && docker build -t bio-api .

# Frontend needs Node.js environment
# Use Vercel, Netlify, or similar for Next.js
```

### Q: What about scaling?
**A:** 
- API servers can be load balanced
- ChromaDB can handle 1M+ vectors
- Use Redis for distributed caching
- Consider CDN for static assets

## 🐛 Common Issues

### Q: "Cannot find module" errors
**A:** Reinstall dependencies:
```bash
rm -rf node_modules package-lock.json
npm install
```

### Q: Ollama not responding
**A:** 
1. Check if running: `ollama list`
2. Start it: `ollama serve`
3. Pull model: `ollama pull gemma3:12b`
4. Verify: `curl http://localhost:11434/api/tags`

### Q: ChromaDB corruption
**A:** Rebuild the database:
```bash
rm -rf python/vector_db/chroma_db
cd python/vector_db
python setup_chromadb.py
```

## 💭 Conceptual

### Q: Why 8-20 words for completions?
**A:** This range provides:
- Enough context for meaningful suggestions
- Not too long to overwhelm users
- Natural sentence completion length
- Fast generation time

### Q: Why use both vector search and LLM?
**A:** 
- Vector search provides real examples and context
- LLM generates novel, contextual completions
- Combination gives best of both worlds

### Q: Can this be used for other text completion tasks?
**A:** Absolutely! The architecture supports any domain:
1. Replace training data
2. Update prompts in `api_server.py`
3. Retrain models
4. Adjust word count limits as needed

## 📚 Resources

### Q: Where can I learn more?
**A:** 
- Architecture: See `ARCHITECTURE.md`
- Training: See `TRAINING_GUIDE.md`
- API Details: See `API_REFERENCE.md`
- Troubleshooting: See `TROUBLESHOOTING.md`

### Q: How do I contribute?
**A:** 
1. Check `DEVELOPMENT_GUIDE.md`
2. Fork the repository
3. Create a feature branch
4. Submit a pull request
5. Ensure all checks pass

## ❓ Still Have Questions?

If your question isn't answered here:
1. Check the documentation in `/Docs`
2. Search existing issues on GitHub
3. Ask in discussions
4. Create a new issue with details

Remember: No question is too simple - we're here to help!