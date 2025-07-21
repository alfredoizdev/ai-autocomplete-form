from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import sys
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add MLX imports
from mlx_lm import load, generate

# Thread pool for running MLX inference
executor = ThreadPoolExecutor(max_workers=1)

# Global model and tokenizer
model = None
tokenizer = None

# MLX Model configuration
MODEL_PATH = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
ADAPTER_PATH = str(Path(__file__).parent.parent / "mlx_training" / "adapters" / "bio_mistral_fresh")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown"""
    # Startup
    global model, tokenizer
    try:
        print(f"Loading MLX model from {MODEL_PATH}")
        print(f"Using adapter: {ADAPTER_PATH}")
        start_time = time.time()
        model, tokenizer = load(MODEL_PATH, adapter_path=ADAPTER_PATH)
        print(f"✅ MLX model loaded successfully in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"❌ Failed to load MLX model: {e}")
        # Exit if model fails to load
        sys.exit(1)
    
    yield
    
    # Shutdown
    print("Application shutting down...")
    executor.shutdown(wait=True)

# Initialize FastAPI app with lifespan
app = FastAPI(title="MLX Bio Autocomplete API", version="1.0.0", lifespan=lifespan)

# Configure CORS for Next.js integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class AutocompleteRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 120
    temperature: Optional[float] = 0.7

class AutocompleteResponse(BaseModel):
    completion: str
    suggestions: List[str]
    elapsed_ms: float
    method: str
    tokens_per_sec: float

class HybridAutocompleteResponse(BaseModel):
    exact_matches: List[str]
    llm_completions: List[str]
    combined_suggestions: List[str]
    elapsed_ms: float
    context_used: bool

def run_mlx_generation(prompt: str, max_tokens: int = 120, min_words: int = 8) -> tuple:
    """Run MLX generation in a thread to avoid blocking"""
    start_time = time.time()
    
    # Format prompt for Mistral instruction model
    full_prompt = f"[INST] Complete the following bio with a full sentence (8-20 words): {prompt} [/INST]"
    
    generated = ""
    attempts = 0
    max_attempts = 3
    
    # Keep generating until we get a completion with enough words
    while attempts < max_attempts:
        # Generate completion with increasing token count each attempt
        current_max_tokens = max_tokens + (attempts * 50)
        response = generate(model, tokenizer, prompt=full_prompt, max_tokens=current_max_tokens)
        
        # Extract generated part
        if "[/INST]" in response:
            generated = response.split("[/INST]")[-1].strip()
        else:
            generated = response
        
        # Check if we have enough words
        word_count = len(generated.split())
        if word_count >= min_words:
            break
            
        attempts += 1
        
        # If still too short, modify prompt to be more explicit
        if attempts == 2:
            full_prompt = f"[INST] Complete the following bio with a FULL SENTENCE that is at least 8 words long: {prompt} [/INST]"
    
    gen_time = time.time() - start_time
    tokens = len(generated.split())
    tokens_per_sec = tokens / gen_time if gen_time > 0 else 0
    
    return generated, gen_time, tokens_per_sec

async def generate_mlx_completion(prompt: str, max_tokens: int = 120) -> dict:
    """Async wrapper for MLX generation"""
    loop = asyncio.get_event_loop()
    generated, gen_time, tokens_per_sec = await loop.run_in_executor(
        executor, run_mlx_generation, prompt, max_tokens
    )
    return {
        "completion": generated,
        "time": gen_time,
        "tokens_per_sec": tokens_per_sec
    }

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "MLX Bio Autocomplete API",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "adapter_path": ADAPTER_PATH
    }

@app.post("/api/autocomplete", response_model=AutocompleteResponse)
async def autocomplete(request: AutocompleteRequest):
    """
    Get autocomplete suggestions using MLX fine-tuned model
    """
    start_time = time.time()
    
    if not model:
        raise HTTPException(
            status_code=503,
            detail="MLX model not loaded"
        )
    
    # Clean the prompt
    prompt = request.prompt.strip()
    
    if not prompt:
        return AutocompleteResponse(
            completion="",
            suggestions=[],
            elapsed_ms=0,
            method="empty_prompt",
            tokens_per_sec=0
        )
    
    try:
        # Generate completion
        result = await generate_mlx_completion(prompt, request.max_tokens)
        
        # Generate a second variant with different sampling
        result2 = await generate_mlx_completion(prompt, request.max_tokens)
        
        # Create suggestions list
        suggestions = [result["completion"]]
        if result2["completion"] != result["completion"]:
            suggestions.append(result2["completion"])
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return AutocompleteResponse(
            completion=result["completion"],
            suggestions=suggestions,
            elapsed_ms=elapsed_ms,
            method="mlx_finetuned",
            tokens_per_sec=result["tokens_per_sec"]
        )
        
    except Exception as e:
        print(f"Error during MLX autocomplete: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.post("/api/autocomplete/hybrid", response_model=HybridAutocompleteResponse)
async def hybrid_autocomplete(request: AutocompleteRequest):
    """
    Hybrid autocomplete using MLX model (compatible with existing API)
    """
    start_time = time.time()
    
    if not model:
        raise HTTPException(
            status_code=503,
            detail="MLX model not loaded"
        )
    
    prompt = request.prompt.strip()
    
    if not prompt:
        return HybridAutocompleteResponse(
            exact_matches=[],
            llm_completions=[],
            combined_suggestions=[],
            elapsed_ms=0,
            context_used=False
        )
    
    try:
        # Generate multiple completions
        tasks = []
        for _ in range(3):  # Generate 3 variants
            tasks.append(generate_mlx_completion(prompt, request.max_tokens))
        
        results = await asyncio.gather(*tasks)
        
        # Extract unique completions
        completions = []
        seen = set()
        for result in results:
            completion = result["completion"]
            if completion and completion.lower() not in seen:
                completions.append(completion)
                seen.add(completion.lower())
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return HybridAutocompleteResponse(
            exact_matches=[],  # MLX model doesn't use exact matches
            llm_completions=completions,
            combined_suggestions=completions[:3],  # Return top 3
            elapsed_ms=elapsed_ms,
            context_used=False  # MLX model has context from training
        )
        
    except Exception as e:
        print(f"Error in MLX hybrid autocomplete: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.get("/api/stats")
async def get_stats():
    """Get statistics about the MLX model"""
    return {
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "adapter_path": ADAPTER_PATH,
        "status": "ready" if model else "not_loaded",
        "model_type": "mlx_finetuned"
    }

if __name__ == "__main__":
    import uvicorn
    
    print("Starting MLX Bio Autocomplete API server...")
    print("API will be available at http://localhost:8003")
    print("API docs available at http://localhost:8003/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        reload=False
    )