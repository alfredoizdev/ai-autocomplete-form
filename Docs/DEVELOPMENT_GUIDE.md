# Development Guide

This guide covers best practices and workflows for developing on the AI Bio Autocomplete project.

## 🛠 Development Setup

### Prerequisites
- Node.js 20.x
- Python 3.10+
- Git
- VS Code (recommended) or your preferred editor

### Initial Setup
```bash
# Clone the repository
git clone <repository-url>
cd ai-train-llm

# Install dependencies
npm install
pip install -r python/requirements.txt

# Set up environment
cp .env.example .env.local
# Edit .env.local with your settings
```

## 📁 Code Organization

### Frontend Structure
```
/app              # Next.js pages (App Router)
/components       # Reusable React components
/hooks           # Custom React hooks
/actions         # Server actions
/lib             # Utility functions
```

### Backend Structure
```
/python/api      # FastAPI servers
/python/vector_db # ChromaDB implementation
/python/mlx_server # MLX model serving
/python/mlx_training # Training scripts
```

## 💻 Development Workflow

### 1. Feature Development

#### Frontend Changes
```bash
# Start dev server
npm run dev

# In another terminal, start backend
./start_hybrid.sh  # or ./start_trained.sh

# Make changes and test
# Hot reload will update the UI
```

#### Backend Changes
```bash
# Python API changes require restart
# Stop the server (Ctrl+C) and restart
./start_hybrid.sh

# For MLX server changes
python python/mlx_server/mlx_model_server.py
```

### 2. Code Style

#### TypeScript/React
- Use TypeScript for all new code
- Follow React 19 best practices
- Use functional components with hooks
- Keep components focused and small

```typescript
// Good: Clear types and single responsibility
interface BioFormProps {
  onSubmit: (bio: string) => void;
  maxLength?: number;
}

export function BioForm({ onSubmit, maxLength = 500 }: BioFormProps) {
  // Component logic
}
```

#### Python
- Follow PEP 8 style guide
- Use type hints for all functions
- Document complex logic

```python
# Good: Type hints and clear naming
def get_autocomplete_suggestions(
    prompt: str, 
    max_suggestions: int = 3
) -> List[str]:
    """Get autocomplete suggestions for the given prompt."""
    # Implementation
```

### 3. Adding New Features

#### Adding a New Hook
1. Create hook file in `/hooks`
2. Follow naming convention: `use<Feature>.tsx`
3. Update `useTextFeatureCoordinator` if needed

```typescript
// hooks/useNewFeature.tsx
export function useNewFeature() {
  const [state, setState] = useState();
  
  // Hook logic
  
  return { state, /* methods */ };
}
```

#### Adding a New API Endpoint
1. Add endpoint to `api_server.py`
2. Create Pydantic models for request/response
3. Update API documentation

```python
@app.post("/api/new-endpoint")
async def new_endpoint(request: NewRequest) -> NewResponse:
    """Document what this endpoint does."""
    # Implementation
```

## 🧪 Testing

### Frontend Testing
```bash
# Run linting
npm run lint

# Build check (catches TypeScript errors)
npm run build

# Manual testing checklist:
# - [ ] Autocomplete works in <150ms
# - [ ] Spell check highlights errors
# - [ ] Kick detection blocks content
# - [ ] Mobile responsive
```

### Backend Testing
```bash
# Test API endpoints
curl -X POST http://localhost:8001/api/autocomplete/hybrid \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Looking for"}'

# Check MLX server
curl http://localhost:8003/health
```

## 🐛 Debugging

### Frontend Debugging
1. Use Chrome DevTools
2. Check Network tab for API calls
3. Console for errors
4. React Developer Tools extension

### Backend Debugging
```python
# Add debug prints
print(f"DEBUG: prompt={prompt}, suggestions={suggestions}")

# Check logs
tail -f /tmp/api_server.log
tail -f /tmp/mlx_server.log
```

### Common Issues
1. **Autocomplete too slow**: Check if caching is working
2. **Wrong suggestions**: Verify ChromaDB has correct data
3. **UI not updating**: Check feature coordinator locks

## 📝 Git Workflow

### Branch Naming
- `feature/add-<feature-name>`
- `fix/repair-<issue>`
- `docs/update-<section>`

### Commit Messages
```bash
# Good commit messages
git commit -m "feat: add grammar filter to training pipeline"
git commit -m "fix: autocomplete returning too many words"
git commit -m "docs: update troubleshooting guide"
```

### Pull Request Process
1. Create feature branch
2. Make changes and test thoroughly
3. Run `npm run lint` and `npm run build`
4. Push branch and create PR
5. Include:
   - What changed and why
   - Testing done
   - Screenshots if UI changes

## 🚀 Performance Considerations

### Frontend
- Use React.memo for expensive components
- Implement proper debouncing (already done in hooks)
- Lazy load heavy components

### Backend
- Cache expensive operations (ChromaDB queries)
- Use async/await properly
- Monitor response times

## 🔧 Configuration

### Environment Variables
```bash
# .env.local
AUTOCOMPLETE_MODE=hybrid  # or 'trained'
OLLAMA_PATH_API=http://127.0.0.1:11434/api
NEXT_PUBLIC_OPENAI_API_KEY=your-key  # optional
```

### Feature Flags
Control features via environment:
```typescript
const ENABLE_SPELL_CHECK = process.env.NEXT_PUBLIC_ENABLE_SPELL_CHECK !== 'false';
```

## 📚 Resources

- [Next.js 15 Docs](https://nextjs.org/docs)
- [React 19 Features](https://react.dev)
- [FastAPI Docs](https://fastapi.tiangolo.com)
- [ChromaDB Guide](https://docs.trychroma.com)
- [MLX Documentation](https://ml-explore.github.io/mlx/)

## 💡 Tips

1. **Keep It Simple**: Don't over-engineer solutions
2. **Test Early**: Verify changes work before large refactors
3. **Document Complex Logic**: Future you will thank you
4. **Ask Questions**: Unclear about something? Check existing code patterns
5. **Performance First**: Keep the 150ms response time target

## 🤝 Contributing

1. Check existing issues/PRs first
2. Discuss large changes before implementing
3. Follow code style guidelines
4. Test thoroughly
5. Update documentation if needed

Remember: The goal is fast, accurate bio completions that help users express themselves!