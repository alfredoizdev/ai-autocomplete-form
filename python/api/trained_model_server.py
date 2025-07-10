"""
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
