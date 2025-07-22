"""
MLX Model Server for bio autocomplete.
Serves the fine-tuned Phi-3 model on port 8003.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import mlx
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from pathlib import Path
import time
import os

app = FastAPI(title="MLX Bio Autocomplete Server", version="1.0.0")

# Enable CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
model = None
tokenizer = None
adapter_path = None

# Request/Response models
class AutocompleteRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 50
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class AutocompleteResponse(BaseModel):
    completion: str
    elapsed_ms: float
    model_name: str

def strip_prompt_from_response(prompt: str, response: str) -> str:
    """Remove the prompt from the beginning of the response."""
    if not prompt or not response:
        return response
    
    # Clean the response
    response = response.strip()
    
    # Remove prompt if it appears at the start
    if response.lower().startswith(prompt.lower()):
        response = response[len(prompt):].strip()
    
    # Also remove "completion:" marker if present
    if response.startswith("completion:"):
        response = response[11:].strip()
    
    # Remove end tokens and unknown tokens
    for token in ["<|end|>", "<|endoftext|>", "<unk>", "[UNK]", "</s>", "<pad>"]:
        if token in response:
            response = response.split(token)[0].strip()
    
    # Remove any remaining special tokens
    response = response.replace("<unk>", "").strip()
    
    return response

@app.on_event("startup")
async def load_model():
    """Load the model and adapter on startup."""
    global model, tokenizer, adapter_path
    
    print("Loading MLX model...")
    
    # Check if we have a fine-tuned adapter
    adapter_dir = Path(__file__).parent / "models" / "bio-phi3-lora"
    
    if adapter_dir.exists() and (adapter_dir / "adapters.safetensors").exists():
        print(f"Loading fine-tuned model from {adapter_dir}")
        adapter_path = str(adapter_dir)
        # Load base model with adapter
        model, tokenizer = load("mlx-community/Phi-3-mini-4k-instruct-4bit", 
                               adapter_path=adapter_path)
        print("✅ Fine-tuned model loaded successfully")
    else:
        print("No fine-tuned model found, loading base model")
        # Load base model without adapter
        model, tokenizer = load("mlx-community/Phi-3-mini-4k-instruct-4bit")
        print("✅ Base model loaded successfully")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "MLX Bio Autocomplete Server",
        "model_loaded": model is not None,
        "adapter_loaded": adapter_path is not None
    }

@app.get("/health")
async def health_check():
    """Health check for monitoring."""
    return {"status": "healthy", "model": "mlx-phi3"}

@app.post("/api/autocomplete/mlx", response_model=AutocompleteResponse)
async def autocomplete(request: AutocompleteRequest):
    """Generate autocomplete suggestions using MLX model."""
    
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Format the prompt for the model
        # For fine-tuned model, use the format it was trained on
        if adapter_path:
            formatted_prompt = f"prompt: {request.prompt} completion:"
        else:
            # For base model, use instruction format
            formatted_prompt = f"Complete this text in a natural way: {request.prompt}"
        
        # Generate completion
        # MLX doesn't accept temperature/top_p directly, use sampler
        sampler = make_sampler(
            temp=request.temperature,
            top_p=request.top_p
        )
        
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            max_tokens=request.max_tokens,
            sampler=sampler,
            verbose=False
        )
        
        # Clean up the response
        completion = strip_prompt_from_response(request.prompt, response)
        
        # Additional cleanup for bio completions
        if completion:
            # Remove any trailing ellipsis
            completion = completion.rstrip("...").rstrip("…").strip()
            
            # Ensure proper capitalization based on prompt ending
            if request.prompt.rstrip().endswith((",", ":")):
                completion = completion[0].lower() + completion[1:] if len(completion) > 1 else completion.lower()
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return AutocompleteResponse(
            completion=completion,
            elapsed_ms=elapsed_ms,
            model_name="phi3-mlx" + ("-finetuned" if adapter_path else "-base")
        )
        
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/autocomplete/mlx/batch")
async def batch_autocomplete(prompts: List[str]):
    """Generate multiple completions in batch."""
    
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for prompt in prompts[:5]:  # Limit to 5 prompts
        try:
            response = await autocomplete(AutocompleteRequest(prompt=prompt))
            results.append({
                "prompt": prompt,
                "completion": response.completion
            })
        except:
            results.append({
                "prompt": prompt,
                "completion": ""
            })
    
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    
    print("Starting MLX Bio Autocomplete Server...")
    print("Server will be available at http://localhost:8003")
    print("API docs available at http://localhost:8003/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        reload=False
    )