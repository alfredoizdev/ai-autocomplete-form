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
import re

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
        sys.exit(1)
    
    yield
    
    # Shutdown
    print("Application shutting down...")
    executor.shutdown(wait=True)

# Initialize FastAPI app with lifespan
app = FastAPI(title="MLX Bio Autocomplete API - Sentence Only", version="3.0.0", lifespan=lifespan)

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
    max_tokens: Optional[int] = 20  # Very limited

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

def extract_first_sentence_completion(text: str) -> str:
    """Extract only the first sentence from generated text"""
    # Remove any leading/trailing whitespace
    text = text.strip()
    
    # Find the first sentence ending
    sentence_end_pattern = r'[.!?]'
    match = re.search(sentence_end_pattern, text)
    
    if match:
        # Return text up to and including the first sentence ending
        return text[:match.end()].strip()
    
    # If no sentence ending found, look for natural breaks
    # Check for common phrase endings that could work as stops
    natural_breaks = [
        ', and', ', but', ', so', ', or', ', which', ', who', ', where',
        ' and ', ' but ', ' so ', ' or ', ' for ',
        '...', '–', '—', ';', ':'
    ]
    
    for break_point in natural_breaks:
        if break_point in text:
            # Return text up to the break point
            return text.split(break_point)[0].strip()
    
    # If still too long, just take first 10-15 words
    words = text.split()
    if len(words) > 15:
        return ' '.join(words[:12]) + '...'
    
    return text

def run_mlx_generation_sentence_only(prompt: str, max_tokens: int = 20) -> tuple:
    """Run MLX generation but strictly limit to one sentence"""
    start_time = time.time()
    
    # Create a prompt that explicitly asks for sentence completion
    full_prompt = f"[INST] Complete this with ONLY the rest of the current sentence (maximum 15 words): {prompt} [/INST]"
    
    # Generate with very limited tokens
    response = generate(
        model, 
        tokenizer, 
        prompt=full_prompt, 
        max_tokens=max_tokens  # Very limited
    )
    
    # Extract generated part
    if "[/INST]" in response:
        generated = response.split("[/INST]")[-1].strip()
    else:
        generated = response
    
    # Aggressively limit to first sentence only
    generated = extract_first_sentence_completion(generated)
    
    # Additional safety: if still too long, truncate
    words = generated.split()
    if len(words) > 15:
        generated = ' '.join(words[:12]) + '...'
    
    gen_time = time.time() - start_time
    tokens = len(generated.split())
    tokens_per_sec = tokens / gen_time if gen_time > 0 else 0
    
    return generated, gen_time, tokens_per_sec

async def generate_mlx_completion_sentence(prompt: str, max_tokens: int = 20) -> dict:
    """Async wrapper for sentence-limited MLX generation"""
    loop = asyncio.get_event_loop()
    generated, gen_time, tokens_per_sec = await loop.run_in_executor(
        executor, run_mlx_generation_sentence_only, prompt, max_tokens
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
        "service": "MLX Bio Autocomplete API - Sentence Only v3",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "adapter_path": ADAPTER_PATH,
        "features": "Strict sentence-only completions"
    }

@app.post("/api/autocomplete", response_model=AutocompleteResponse)
async def autocomplete(request: AutocompleteRequest):
    """Get autocomplete suggestions - strictly one sentence"""
    start_time = time.time()
    
    if not model:
        raise HTTPException(
            status_code=503,
            detail="MLX model not loaded"
        )
    
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
        # Generate single completion
        result = await generate_mlx_completion_sentence(prompt, request.max_tokens)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return AutocompleteResponse(
            completion=result["completion"],
            suggestions=[result["completion"]],
            elapsed_ms=elapsed_ms,
            method="mlx_sentence_only",
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
    """Hybrid autocomplete - sentence-only version"""
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
        # Generate 2 variations (not 3, to save time)
        tasks = []
        for _ in range(2):
            tasks.append(generate_mlx_completion_sentence(prompt, request.max_tokens))
        
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
            exact_matches=[],
            llm_completions=completions,
            combined_suggestions=completions,
            elapsed_ms=elapsed_ms,
            context_used=False
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
        "model_type": "mlx_sentence_only_v3",
        "features": {
            "strict_sentence_limit": True,
            "max_words": 15,
            "aggressive_truncation": True,
            "natural_break_detection": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    print("Starting MLX Bio Autocomplete API - Sentence Only v3...")
    print("Features: Strict single sentence completions only")
    print("API will be available at http://localhost:8003")
    print("API docs available at http://localhost:8003/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        reload=False
    )