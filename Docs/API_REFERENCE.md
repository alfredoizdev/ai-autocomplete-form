# API Reference - Backend Services

This guide is for developers who want to understand or modify how the backend works.

## 🤔 What's an API?

An API (Application Programming Interface) is how the frontend (what you see) talks to the backend (the AI brain). Think of it like a waiter taking your order to the kitchen.

When you type in the text box:
1. Frontend sends your text to the API
2. API processes it (searches/generates)
3. API sends back suggestions
4. Frontend shows them as gray text

## 🌐 The Two Backend Services

### 1. Hybrid API Server (Port 8001)
**What it does**: Combines database search with AI generation
- **When it runs**: When you use `./start_hybrid.sh`
- **Interactive docs**: http://localhost:8001/docs
- **Best for**: General use, variety in suggestions

### 2. MLX Model Server (Port 8003)
**What it does**: Uses your custom-trained model for suggestions
- **When it runs**: When you use `./start_trained.sh`
- **Requires**: A trained model (see Training Guide)
- **Best for**: Consistent style, faster responses

## 📡 Main API Endpoints

### Testing if Services are Running

**Check Hybrid API:**
```bash
curl http://localhost:8001/
# Should return: {"status":"ok"}
```

**Check MLX Server:**
```bash
curl http://localhost:8003/
# Should return: {"status":"ok","model_loaded":true}
```

### The Main Endpoints You'll Use

#### Hybrid Mode - Get Suggestions
**Endpoint**: `POST http://localhost:8001/api/autocomplete/hybrid`

**What you send:**
```json
{
  "prompt": "We are a fun couple who"
}
```

**What you get back:**
```json
{
  "combined_suggestions": [
    "enjoys meeting new people for friendship and fun",
    "likes to explore new experiences together"
  ],
  "elapsed_ms": 145.23
}
```

#### Trained Mode - Get Suggestions
**Endpoint**: `POST http://localhost:8003/api/autocomplete/mlx`

**What you send:**
```json
{
  "prompt": "Looking for couples who"
}
```

**What you get back:**
```json
{
  "completion": "enjoy dinners, dancing, and good conversation.",
  "elapsed_ms": 98.76
}
```

### 🎮 Try It Yourself!

With the backend running, you can test these endpoints:

**Test Hybrid Mode:**
```bash
curl -X POST http://localhost:8001/api/autocomplete/hybrid \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I am looking for"}'
```

**Test Trained Mode:**
```bash
curl -X POST http://localhost:8003/api/autocomplete/mlx \
  -H "Content-Type: application/json" \
  -d '{"prompt": "We enjoy"}'
```

## 🔧 How the Frontend Uses These APIs

The app automatically picks the right API based on which mode you're running:

- **Hybrid mode** → Uses port 8001
- **Trained mode** → Uses port 8003

This happens automatically when you use the startup scripts!

## ⚙️ Customizing API Behavior

### Want Different Suggestions?

For **Hybrid Mode**, you can adjust these settings in `python/api/api_server.py`:
- `NUM_SIMILAR_BIOS = 10` - How many similar examples to find
- `MIN_SUGGESTION_LENGTH = 8` - Minimum words in suggestions

For **Trained Mode**, the behavior is determined by your trained model.

### Response Speed vs Quality

The app is optimized for the best balance, but you can adjust:
- **Faster**: Reduce the number of suggestions generated
- **Better**: Increase the AI "temperature" for more creative responses

Details in the [Configuration Guide](./CONFIGURATION.md).

## 🚨 Understanding API Errors

If something goes wrong, the API returns clear error messages:

**Success (200)**: Everything worked!
```json
{"combined_suggestions": ["..."], "elapsed_ms": 123}
```

**Server Error (500)**: Something broke on our end
```json
{"detail": "Model failed to generate response"}
```

**Service Unavailable (503)**: Backend not ready yet
```json
{"detail": "Model still loading, please wait"}
```

## 🧪 API Playground

Want to explore the APIs interactively?

1. Make sure Hybrid mode is running
2. Open http://localhost:8001/docs
3. Try out the endpoints with the "Try it out" button!

This interactive documentation lets you:
- Test endpoints without writing code
- See all available options
- Understand request/response formats

## 📊 Performance Tips

### Making APIs Faster
1. **Use caching** - The frontend already does this
2. **Batch requests** - Send multiple prompts at once (future feature)
3. **Optimize prompts** - Shorter prompts = faster responses

### Monitoring Performance
Watch the logs to see response times:
```bash
# See each request and how long it took
tail -f python/api_server.log
```

Look for lines like:
```
INFO: Autocomplete request processed in 145.23ms
```

## 🚀 Advanced Usage

### Building Your Own Frontend?

The APIs are standard REST endpoints, so you can use them from:
- React/Vue/Angular apps
- Mobile apps
- Command line tools
- Any programming language!

Example in JavaScript:
```javascript
const response = await fetch('http://localhost:8001/api/autocomplete/hybrid', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ prompt: 'We are looking for' })
});
const data = await response.json();
console.log(data.combined_suggestions);
```

## 📚 Next Steps

- **Just want to use the app?** You don't need to worry about APIs!
- **Building something new?** Use the playground at http://localhost:8001/docs
- **Having issues?** Check the [Troubleshooting Guide](./TROUBLESHOOTING.md)

---

*Remember: APIs are just how different parts of the app talk to each other. The startup scripts handle all the complexity for you!*