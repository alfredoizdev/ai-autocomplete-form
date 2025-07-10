# Integration Instructions

## 1. Start the Trained Model Server

```bash
cd python
source venv/bin/activate
uvicorn api.trained_model_server:app --reload --port 8002
```

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
    
    // Try trained model
    const trainedResponse = await fetch('http://localhost:8002/api/autocomplete/trained', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: input, max_suggestions: 2 })
    });
    
    const trainedSuggestions = trainedResponse.ok
      ? (await trainedResponse.json()).suggestions
      : [];
    
    // Combine suggestions
    const allSuggestions = [...vectorSuggestions];
    trainedSuggestions.forEach(s => {
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
   - Trained model server on port 8002
   - Ollama on port 11434
   - Next.js app on port 3000

2. Test autocomplete with various prompts

## Benefits of this Approach:

1. **Fast Response**: Vector search provides quick exact matches
2. **Creative Completions**: Trained model adds novel suggestions
3. **Fallback Support**: If one service fails, the other still works
4. **Best of Both Worlds**: Combines accuracy with creativity
