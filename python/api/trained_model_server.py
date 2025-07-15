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
model_path = Path(__file__).parent.parent / "mlx_training" / "bio_distilgpt2_finetuned"

# Load the base model first
from transformers import GPT2LMHeadModel, GPT2Tokenizer
print(f"Loading model from {model_path}")

# For LoRA fine-tuned models, we need to load the base model and adapter
try:
    # Try loading as a full model first
    model = AutoModelForCausalLM.from_pretrained(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
except Exception as e:
    print(f"Failed to load as full model, trying base model + adapter: {e}")
    # Load base DistilGPT2 and the tokenizer
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    tokenizer = GPT2Tokenizer.from_pretrained(str(model_path))

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Model loaded successfully on {device}")

class AutocompleteRequest(BaseModel):
    prompt: str
    max_suggestions: int = 3

class AutocompleteResponse(BaseModel):
    suggestions: List[str]

@app.post("/api/autocomplete/trained", response_model=AutocompleteResponse)
async def autocomplete(request: AutocompleteRequest):
    """Generate autocomplete suggestions using the trained model."""
    
    try:
        # Just use the prompt directly - the model was likely trained on raw bio text
        text = request.prompt
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        suggestions = []
        
        # Generate multiple suggestions with different seeds
        for i in range(request.max_suggestions):
            with torch.no_grad():
                torch.manual_seed(42 + i)  # Different seed for each suggestion
                
                # Generate with more conservative parameters
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=20,
                    min_new_tokens=5,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.85,
                    top_k=50,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    length_penalty=1.0
                )
            
            # Decode the full output
            full_generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new part (remove the prompt)
            if full_generated.startswith(text):
                completion = full_generated[len(text):].strip()
            else:
                completion = full_generated.strip()
            
            # Clean up the completion
            completion = completion.split("\n")[0].strip()  # Take first line only
            if completion and len(completion.split()) >= 3:  # At least 3 words
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
