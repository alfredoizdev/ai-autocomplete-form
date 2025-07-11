from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import sys
from pathlib import Path
import httpx
import os
from typing import List, Optional

# Add parent directory to path to import vector_db module
sys.path.append(str(Path(__file__).parent.parent))

from vector_db.vector_search import BioVectorSearch

# Initialize FastAPI app
app = FastAPI(title="Bio Autocomplete API", version="1.0.0")

# Configure CORS for Next.js integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector search (will be done on startup)
vector_search = None

# Ollama configuration
OLLAMA_API_URL = os.getenv("OLLAMA_PATH_API", "http://127.0.0.1:11434/api")

# Request/Response models
class AutocompleteRequest(BaseModel):
    prompt: str

class AutocompleteResponse(BaseModel):
    completion: str
    suggestions: list[str]
    elapsed_ms: float
    method: str

class HybridAutocompleteResponse(BaseModel):
    exact_matches: List[str]
    llm_completions: List[str]
    combined_suggestions: List[str]
    elapsed_ms: float
    context_used: bool

@app.on_event("startup")
async def startup_event():
    """Initialize vector search on startup"""
    global vector_search
    try:
        print("Initializing vector search...")
        vector_search = BioVectorSearch()
        print("Vector search initialized successfully")
    except Exception as e:
        print(f"Failed to initialize vector search: {e}")
        # Don't exit - allow server to start even if vector search fails

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Bio Autocomplete API",
        "vector_search_ready": vector_search is not None
    }

@app.post("/api/autocomplete", response_model=AutocompleteResponse)
async def autocomplete(request: AutocompleteRequest):
    """
    Get autocomplete suggestions for the given prompt
    """
    start_time = time.time()
    
    if not vector_search:
        raise HTTPException(
            status_code=503,
            detail="Vector search not initialized"
        )
    
    # Clean the prompt
    prompt = request.prompt.strip()
    
    if not prompt:
        return AutocompleteResponse(
            completion="",
            suggestions=[],
            elapsed_ms=0,
            method="empty_prompt"
        )
    
    try:
        # Get suggestions from vector search
        suggestions = vector_search.get_autocomplete_suggestions(
            prompt,
            max_suggestions=3
        )
        
        # Use the first suggestion as the main completion
        completion = suggestions[0] if suggestions else ""
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return AutocompleteResponse(
            completion=completion,
            suggestions=suggestions,
            elapsed_ms=elapsed_ms,
            method="vector_search"
        )
        
    except Exception as e:
        print(f"Error during autocomplete: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.get("/api/stats")
async def get_stats():
    """Get statistics about the vector database"""
    if not vector_search:
        return {"error": "Vector search not initialized"}
    
    try:
        collection_count = vector_search.collection.count()
        return {
            "total_bios": collection_count,
            "status": "ready",
            "embedding_model": "default"
        }
    except Exception as e:
        return {"error": str(e)}

async def call_ollama_with_context(prompt: str, contexts: List[str]) -> List[str]:
    """
    Call Ollama API with context for better completions
    """
    # Format contexts into a single string
    context_text = "\n\n".join(contexts) if contexts else ""
    
    messages = [
        {
            "role": "system",
            "content": """You write seductive bio completions that make swingers desperately want to meet and play.

Write like you're seducing someone into your bed. Make them feel desired, excited, and eager to explore.

CRITICAL: Always complete the thought in a way that:
- Creates anticipation and desire
- Makes them imagine the pleasure ahead
- Sounds confident, playful, and irresistibly naughty
- Forms a complete, enticing sentence (8-20 words)
- Uses explicit language about pleasure, orgasms, and desires

Examples of seductive completions:
"I am a young male swinger looking for" → "sexy couples who want to explore every inch of pleasure together tonight"
"We love meeting" → "adventurous souls ready to turn our wildest fantasies into unforgettable orgasmic reality"
"My biggest turn on is" → "watching my partner moan with pleasure while we share you completely"
"Looking for couples who" → "crave intense oral pleasure and aren't afraid to get deliciously dirty with us"

Be explicitly sexual. Make them want to message immediately. Output ONLY the seductive completion."""
        }
    ]
    
    # Add context if available
    if context_text:
        messages.append({
            "role": "user",
            "content": f"Here are some example bios for context:\n{context_text}"
        })
    
    messages.append({
        "role": "user",
        "content": f"Complete this bio text: {prompt}"
    })
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{OLLAMA_API_URL}/chat",
                json={
                    "model": "gemma3:12b",
                    "messages": messages,
                    "stream": False,
                    "temperature": 0.85,
                    "top_p": 0.95,
                    "max_tokens": 100,
                    "stop": ["\n", "\n\n"]
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("message", {}).get("content", "").strip()
                
                # Clean up the response
                content = content.replace("...", "").replace("…", "").strip()
                
                # Post-process: ensure lowercase if input ends with comma or no sentence-ending punctuation
                if content and prompt:
                    last_char = prompt.rstrip()[-1] if prompt.rstrip() else ""
                    if last_char in [",", ":", ";"] or (last_char and last_char not in [".", "!", "?"]):
                        # Force lowercase on first character
                        content = content[0].lower() + content[1:] if len(content) > 1 else content.lower()
                
                # Quality filter: ensure completion is meaningful and complete
                if content and len(content.split()) >= 8:  # Minimum 8 words for complete thought
                    completions = [content]
                else:
                    completions = []
                
                # Try to get one more variation with slightly different temperature
                if content:
                    response2 = await client.post(
                        f"{OLLAMA_API_URL}/chat",
                        json={
                            "model": "gemma3:12b",
                            "messages": messages,
                            "stream": False,
                            "temperature": 0.9,
                            "top_p": 0.95,
                            "max_tokens": 100,
                            "stop": ["\n", "\n\n"]
                        }
                    )
                    
                    if response2.status_code == 200:
                        data2 = response2.json()
                        content2 = data2.get("message", {}).get("content", "").strip()
                        content2 = content2.replace("...", "").replace("…", "").strip()
                        
                        # Apply same post-processing for capitalization
                        if content2 and prompt:
                            last_char = prompt.rstrip()[-1] if prompt.rstrip() else ""
                            if last_char in [",", ":", ";"] or (last_char and last_char not in [".", "!", "?"]):
                                content2 = content2[0].lower() + content2[1:] if len(content2) > 1 else content2.lower()
                        
                        # Quality filter for second completion too
                        if content2 and content2 != content and len(content2.split()) >= 8:
                            completions.append(content2)
                
                return completions
            else:
                print(f"Ollama API error: {response.status_code}")
                return []
                
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return []

@app.post("/api/autocomplete/hybrid", response_model=HybridAutocompleteResponse)
async def hybrid_autocomplete(request: AutocompleteRequest):
    """
    Hybrid autocomplete that combines vector search with LLM generation
    """
    start_time = time.time()
    
    if not vector_search:
        raise HTTPException(
            status_code=503,
            detail="Vector search not initialized"
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
        # Get both contexts and exact matches from vector search
        vector_results = vector_search.get_context_and_suggestions(
            prompt,
            n_contexts=3,
            max_suggestions=2
        )
        
        contexts = vector_results.get('contexts', [])
        exact_matches = vector_results.get('suggestions', [])
        
        # Get LLM completions with context
        llm_completions = await call_ollama_with_context(prompt, contexts)
        
        # Combine suggestions, removing duplicates
        combined = []
        seen = set()
        
        # Add exact matches first (but filter out poor quality ones)
        incomplete_endings = [" in", " for", " with", " at", " to", " of", " the", " a", " an", " or", " and", " but"]
        for suggestion in exact_matches:
            # Skip suggestions that are fragments or don't make sense
            suggestion_lower = suggestion.lower().strip()
            is_incomplete = any(suggestion_lower.endswith(ending) for ending in incomplete_endings)
            
            if (suggestion_lower not in seen and 
                len(suggestion.split()) >= 5 and  # Increased minimum
                not suggestion.startswith(", which") and
                not suggestion.endswith(" is") and
                not is_incomplete and
                suggestion.strip()[-1] not in [',', ':']):  # Must end properly
                combined.append(suggestion)
                seen.add(suggestion_lower)
        
        # Add LLM completions
        for completion in llm_completions:
            if completion.lower() not in seen and len(combined) < 5:
                combined.append(completion)
                seen.add(completion.lower())
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return HybridAutocompleteResponse(
            exact_matches=exact_matches,
            llm_completions=llm_completions,
            combined_suggestions=combined[:3],  # Return top 3
            elapsed_ms=elapsed_ms,
            context_used=bool(contexts)
        )
        
    except Exception as e:
        print(f"Error in hybrid autocomplete: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Bio Autocomplete API server...")
    print("API will be available at http://localhost:8001")
    print("API docs available at http://localhost:8001/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False  # Disable reload to avoid import string requirement
    )