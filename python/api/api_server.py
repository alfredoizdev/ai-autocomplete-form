from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import sys
from pathlib import Path

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

# Request/Response models
class AutocompleteRequest(BaseModel):
    prompt: str

class AutocompleteResponse(BaseModel):
    completion: str
    suggestions: list[str]
    elapsed_ms: float
    method: str

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

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Bio Autocomplete API server...")
    print("API will be available at http://localhost:8001")
    print("API docs available at http://localhost:8001/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=True
    )