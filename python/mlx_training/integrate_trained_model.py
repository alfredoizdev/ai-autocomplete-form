"""
Integrate the trained GPT2 model with the autocomplete system.
Since GGUF conversion for GPT2 is complex, we'll use the model directly.
"""

import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def create_inference_server():
    """Create a simple FastAPI server script for the trained model."""
    
    server_code = '''"""
FastAPI server for the trained bio autocomplete model.
Run with: uvicorn trained_model_server:app --reload --port 8002
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
from typing import List

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model_path = Path(__file__).parent / "mlx_training" / "trained_small_model"
model = AutoModelForCausalLM.from_pretrained(str(model_path))
tokenizer = AutoTokenizer.from_pretrained(str(model_path))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

class AutocompleteRequest(BaseModel):
    prompt: str
    max_suggestions: int = 3

class AutocompleteResponse(BaseModel):
    suggestions: List[str]

@app.post("/api/autocomplete/trained", response_model=AutocompleteResponse)
async def autocomplete(request: AutocompleteRequest):
    """Generate autocomplete suggestions using the trained model."""
    
    try:
        # Format prompt
        text = f"Bio: {request.prompt} ->"
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        suggestions = []
        
        # Generate multiple suggestions with different seeds
        for i in range(request.max_suggestions):
            with torch.no_grad():
                torch.manual_seed(42 + i)  # Different seed for each suggestion
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=15,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract completion
            if "->" in generated:
                completion = generated.split("->")[-1].strip()
                # Clean up the completion
                completion = completion.split(".")[0].strip()
                if completion and len(completion) > 2:
                    suggestions.append(completion)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique_suggestions.append(s)
        
        return AutocompleteResponse(suggestions=unique_suggestions[:request.max_suggestions])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "trained_gpt2"}
'''
    
    server_path = Path(__file__).parent.parent / "api" / "trained_model_server.py"
    with open(server_path, 'w') as f:
        f.write(server_code)
    
    print(f"Created server script at: {server_path}")
    return server_path

def create_hybrid_integration():
    """Create a script that combines vector search with the trained model."""
    
    hybrid_code = '''"""
Hybrid autocomplete: Combines vector search with trained model.
"""

import asyncio
from typing import List
import aiohttp

async def get_vector_suggestions(prompt: str) -> List[str]:
    """Get suggestions from vector search."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:8001/api/autocomplete',
                json={'prompt': prompt}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('suggestions', [])
    except:
        return []

async def get_trained_model_suggestions(prompt: str) -> List[str]:
    """Get suggestions from trained model."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:8002/api/autocomplete/trained',
                json={'prompt': prompt, 'max_suggestions': 2}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('suggestions', [])
    except:
        return []

async def get_hybrid_suggestions(prompt: str) -> List[str]:
    """Combine suggestions from both sources."""
    # Get suggestions from both sources in parallel
    vector_task = get_vector_suggestions(prompt)
    model_task = get_trained_model_suggestions(prompt)
    
    vector_suggestions, model_suggestions = await asyncio.gather(
        vector_task, model_task
    )
    
    # Combine and deduplicate
    all_suggestions = []
    
    # Add vector suggestions first (they're usually more accurate for exact matches)
    all_suggestions.extend(vector_suggestions[:2])
    
    # Add model suggestions
    for suggestion in model_suggestions:
        if suggestion not in all_suggestions:
            all_suggestions.append(suggestion)
    
    return all_suggestions[:3]  # Return top 3

if __name__ == "__main__":
    # Test the hybrid approach
    test_prompts = [
        "We are a couple who",
        "Looking for friends to",
        "My hobbies include"
    ]
    
    for prompt in test_prompts:
        suggestions = asyncio.run(get_hybrid_suggestions(prompt))
        print(f"\\nPrompt: '{prompt}'")
        print(f"Suggestions: {suggestions}")
'''
    
    hybrid_path = Path(__file__).parent.parent / "api" / "hybrid_autocomplete.py"
    with open(hybrid_path, 'w') as f:
        f.write(hybrid_code)
    
    print(f"Created hybrid integration at: {hybrid_path}")
    return hybrid_path

def update_typescript_action():
    """Create an updated TypeScript action that uses the hybrid approach."""
    
    updated_action = '''// Updated ai-text.ts to use hybrid approach
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
'''
    
    update_path = Path(__file__).parent.parent / "integration_instructions.md"
    with open(update_path, 'w') as f:
        f.write(f"""# Integration Instructions

## 1. Start the Trained Model Server

```bash
cd python
source venv/bin/activate
uvicorn api.trained_model_server:app --reload --port 8002
```

## 2. Update Your TypeScript Code

Add this to your `actions/ai-text.ts`:

```typescript
{updated_action}
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
""")
    
    print(f"Created integration instructions at: {update_path}")
    return update_path

def main():
    """Main integration function."""
    print("=== Integrating Trained Model ===")
    
    # Check if model exists
    model_path = Path(__file__).parent / "trained_small_model"
    if not model_path.exists():
        print("❌ Trained model not found!")
        return
    
    print("✅ Found trained model")
    
    # Create integration components
    print("\nCreating integration components...")
    
    server_path = create_inference_server()
    hybrid_path = create_hybrid_integration()
    instructions_path = update_typescript_action()
    
    print("\n" + "="*50)
    print("✅ Integration components created!")
    print("="*50)
    
    print("\nNext steps:")
    print("1. Start the trained model server:")
    print("   cd python && source venv/bin/activate")
    print("   uvicorn api.trained_model_server:app --reload --port 8002")
    print("\n2. Follow the integration instructions in:")
    print(f"   {instructions_path}")
    print("\n3. Test the hybrid autocomplete system")

if __name__ == "__main__":
    main()