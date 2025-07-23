# Integration Instructions

## 1. Start the MLX Model Server

```bash
cd python/mlx_server
python mlx_model_server.py
```

This starts the MLX model server on port 8003 with fine-tuned Llama models.

## 2. Update Your TypeScript Code

Add this to your `actions/ai-text.ts`:

```typescript
// Updated ai-text.ts to use hybrid approach
// Add this function to your existing ai-text.ts file

export async function getHybridAutocomplete(input: string): Promise<string[]> {
  try {
    // Try vector search first
    const vectorResponse = await fetch('http://localhost:8001/api/autocomplete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: input })
    });
    
    const vectorSuggestions = vectorResponse.ok 
      ? (await vectorResponse.json()).suggestions 
      : [];
    
    // Try MLX model
    const mlxResponse = await fetch('http://localhost:8003/api/autocomplete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: input, max_suggestions: 2 })
    });
    
    const mlxSuggestions = mlxResponse.ok
      ? (await mlxResponse.json()).suggestions
      : [];
    
    // Combine suggestions
    const allSuggestions = [...vectorSuggestions];
    mlxSuggestions.forEach(s => {
      if (!allSuggestions.includes(s)) {
        allSuggestions.push(s);
      }
    });
    
    return allSuggestions.slice(0, 3);
  } catch (error) {
    console.error('Hybrid autocomplete error:', error);
    return [];
  }
}

```

## 3. Update Your Form Component

In your form component, update the autocomplete to use the hybrid approach:

```typescript
// Replace the existing autocomplete call with:
const suggestions = await getHybridAutocomplete(inputText);
```

## 4. Test the Integration

1. Make sure all services are running:
   - Vector search server on port 8001
   - MLX model server on port 8003
   - Ollama on port 11434
   - Next.js app on port 3000

2. Test autocomplete with various prompts

## Benefits of this Approach:

1. **Fast Response**: Vector search provides quick exact matches
2. **Creative Completions**: MLX fine-tuned models add novel suggestions
3. **Fallback Support**: If one service fails, the other still works
4. **Best of Both Worlds**: Combines accuracy with creativity