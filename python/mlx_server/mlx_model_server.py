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
    stop: Optional[List[str]] = None

class AutocompleteResponse(BaseModel):
    completion: str
    elapsed_ms: float
    model_name: str

def strip_prompt_from_response(prompt: str, response: str) -> str:
    """Remove the prompt from the beginning of the response and fix grammar."""
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
    
    # Fix common grammar issues
    response = fix_grammar_issues(prompt, response)
    
    return response

def fix_final_capitalization(completion: str, prompt: str) -> str:
    """Fix capitalization issues in the final completion."""
    import re
    
    # Fix standalone lowercase 'i' to 'I'
    completion = re.sub(r'\bi\b', 'I', completion)
    
    # Fix "i'm", "i'll", "i've", "i'd" etc.
    completion = re.sub(r'\bi\'', 'I\'', completion)
    
    # Only lowercase first letter if prompt ends with comma/colon AND 
    # the completion starts with a continuation word
    if prompt.rstrip().endswith((",", ":")):
        words = completion.split()
        if words:
            first_word = words[0].lower()
            continuation_words = {'and', 'but', 'or', 'who', 'which', 'that', 'where', 'when', 
                                 'with', 'to', 'for', 'in', 'about', 'from'}
            if first_word in continuation_words:
                completion = completion[0].lower() + completion[1:] if len(completion) > 1 else completion
    
    return completion

def fix_grammar_issues(prompt: str, completion: str) -> str:
    """Fix common grammar issues in completions."""
    print(f"DEBUG fix_grammar_issues called with prompt='{prompt}', completion='{completion}'", flush=True)
    if not prompt or not completion:
        return completion
    
    prompt_lower = prompt.strip().lower()
    completion_lower = completion.strip().lower()
    
    # Fix incomplete adjective phrases like "well hung" missing a noun
    if prompt_lower.endswith("well hung"):
        # Add appropriate noun if completion doesn't start with one
        if not any(completion_lower.startswith(noun) for noun in ["man", "guy", "male", "person", "gentleman"]):
            completion = "man " + completion
    elif prompt_lower.endswith("gentle well hung"):
        if not any(completion_lower.startswith(noun) for noun in ["man", "guy", "male", "person", "gentleman"]):
            completion = "man " + completion
    
    # Fix "looking for to [verb]" pattern
    if prompt_lower.endswith("looking for") and completion.strip().startswith("to "):
        # Common patterns to insert
        if "have" in completion[:20]:
            completion = "people " + completion
        elif "meet" in completion[:20]:
            completion = "someone " + completion
        elif "enjoy" in completion[:20]:
            completion = "ways " + completion
        else:
            completion = "someone " + completion
    
    # Fix "looking for that" pattern - determine if we need "someone" or "something"
    if prompt_lower.endswith("looking for") and completion_lower.startswith("that "):
        # Check if we're talking about people based on context
        person_indicators = ['male', 'female', 'man', 'woman', 'couple', 'person', 'people', 
                           'guy', 'girl', 'lady', 'gentleman', 'swinger', 'partner', 
                           'lover', 'friend', 'mate', 'date', 'companion']
        
        # Check if the prompt or completion indicates we're talking about people
        is_about_people = any(indicator in prompt_lower for indicator in person_indicators)
        
        # Also check the completion for person-related verbs/phrases
        person_verbs = ['likes', 'loves', 'enjoys', 'wants', 'shares', 'understands', 
                       'appreciates', 'knows', 'believes', 'thinks', 'feels']
        if any(verb in completion_lower[:50] for verb in person_verbs):
            is_about_people = True
        
        # Use "someone" for people, "something" for things
        if is_about_people:
            completion = "someone " + completion
        else:
            completion = "something " + completion
    
    # Fix "looking for who" pattern - should be "looking for someone who"
    if prompt_lower.endswith("looking for") and completion_lower.startswith("who "):
        completion = "someone " + completion
    
    # Fix awkward "couple looking for" patterns
    if "couple looking for" in prompt_lower:
        if completion.strip().startswith("to "):
            completion = "couples " + completion
        elif completion_lower.startswith("that "):
            # Check if it's already talking about people or needs "couples"
            # Don't prepend if the completion already makes sense
            person_verbs = ['likes', 'loves', 'enjoys', 'wants', 'shares', 'understands']
            if any(verb in completion_lower[:50] for verb in person_verbs):
                # It's describing people, use "someone" or "people"
                completion = "people " + completion
            else:
                # It might be describing activities or things
                completion = "couples " + completion
        elif completion_lower.startswith("who "):
            # For couples, use "people who" instead of "someone who"
            completion = "people " + completion
    
    # Remove duplicate "someone" if it was added as guidance
    if prompt_lower.endswith("looking for") and completion.startswith("someone someone"):
        completion = completion[8:].strip()
    
    # Ensure sentences end properly
    print(f"DEBUG: Before punctuation check, completion='{completion}'", flush=True)
    if completion and not completion.rstrip().endswith((".", "!", "?", "...", "…")):
        # Check if we have a complete thought
        words = completion.split()
        if len(words) >= 3:  # Lowered threshold for bio completions
            # Add period if it seems like a complete thought
            last_word = words[-1].lower().rstrip(",;:")
            # Don't add period after certain words that suggest incompleteness
            incomplete_endings = ["and", "or", "but", "with", "for", "to", "of", "in", "on", "at", "the", "a", "an"]
            
            # DEBUG
            print(f"DEBUG: Checking completion: '{completion}'", flush=True)
            print(f"DEBUG: Last word: '{last_word}'", flush=True)
            print(f"DEBUG: In incomplete endings: {last_word in incomplete_endings}", flush=True)
            
            if last_word not in incomplete_endings:
                # Check if the last few words form a complete phrase
                if len(words) >= 2:
                    last_two_words = " ".join(words[-2:]).lower()
                    # Common complete phrase endings in bio context
                    complete_phrases = ["have fun", "good time", "new friends", "and see", "is possible", 
                                      "the bedroom", "and friendship", "new experiences", "our fantasy", 
                                      "adult fun", "and play", "great time", "looking for", "interested in",
                                      "more experience", "watch and see", "intimate experience", "and explore",
                                      "new things", "and enjoy", "have some fun", "and chat", "and relax"]
                    if any(last_two_words.endswith(phrase) for phrase in complete_phrases):
                        completion += "."
                    elif last_word not in incomplete_endings:
                        # Add period for most other cases - bio completions should be complete thoughts
                        completion += "."
                elif last_word not in incomplete_endings:
                    # Single or two word completions that seem complete
                    completion += "."
    
    return completion

@app.on_event("startup")
async def load_model():
    """Load the model and adapter on startup."""
    global model, tokenizer, adapter_path
    
    print("Loading MLX model...")
    
    # Check if we have a fine-tuned adapter
    # First check for the new high-quality 3B LookingFor model
    lookingfor_3b_hq_adapter_dir = Path(__file__).parent.parent.parent / "models" / "lookingfor-llama3-3b-hq-lora"
    lookingfor_3b_adapter_dir = Path(__file__).parent.parent.parent / "models" / "lookingfor-llama3-3b-lora"
    lookingfor_1b_adapter_dir = Path(__file__).parent.parent.parent / "models" / "lookingfor-llama3-lora"
    llama_continued_dir = Path(__file__).parent.parent.parent / "models" / "bio-sentence-llama3-lora-continued"
    llama_adapter_dir = Path(__file__).parent.parent.parent / "models" / "bio-sentence-llama3-lora"
    phi_adapter_dir = Path(__file__).parent / "models" / "bio-phi3-lora"
    
    if lookingfor_3b_hq_adapter_dir.exists() and (lookingfor_3b_hq_adapter_dir / "adapters.safetensors").exists():
        print(f"Loading HIGH-QUALITY LookingFor Llama-3.2-3B model from {lookingfor_3b_hq_adapter_dir}")
        adapter_path = str(lookingfor_3b_hq_adapter_dir)
        # Load Llama 3B base model with adapter
        model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit", 
                               adapter_path=adapter_path)
        print("✅ HIGH-QUALITY LookingFor Llama-3.2-3B model loaded successfully (2000 iterations, grammar-filtered)")
    elif lookingfor_3b_adapter_dir.exists() and (lookingfor_3b_adapter_dir / "adapters.safetensors").exists():
        print(f"Loading LookingFor Llama-3.2-3B model from {lookingfor_3b_adapter_dir}")
        adapter_path = str(lookingfor_3b_adapter_dir)
        # Load Llama 3B base model with adapter
        model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit", 
                               adapter_path=adapter_path)
        print("✅ LookingFor Llama-3.2-3B model loaded successfully")
    elif lookingfor_1b_adapter_dir.exists() and (lookingfor_1b_adapter_dir / "adapters.safetensors").exists():
        print(f"Loading LookingFor Llama-3.2-1B model from {lookingfor_1b_adapter_dir}")
        adapter_path = str(lookingfor_1b_adapter_dir)
        # Load Llama 1B base model with adapter
        model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit", 
                               adapter_path=adapter_path)
        print("✅ LookingFor Llama-3.2-1B model loaded successfully")
    elif llama_continued_dir.exists() and (llama_continued_dir / "adapters.safetensors").exists():
        print(f"Loading continued training Llama-3.2-3B model from {llama_continued_dir}")
        adapter_path = str(llama_continued_dir)
        # Load Llama base model with adapter
        model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit", 
                               adapter_path=adapter_path)
        print("✅ Continued training Llama-3.2-3B model (1000 iterations) loaded successfully")
    elif llama_adapter_dir.exists() and (llama_adapter_dir / "adapters.safetensors").exists():
        print(f"Loading fine-tuned Llama-3.2-3B model from {llama_adapter_dir}")
        adapter_path = str(llama_adapter_dir)
        # Load Llama base model with adapter
        model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit", 
                               adapter_path=adapter_path)
        print("✅ Fine-tuned Llama-3.2-3B model loaded successfully")
    elif phi_adapter_dir.exists() and (phi_adapter_dir / "adapters.safetensors").exists():
        print(f"Loading fine-tuned Phi-3 model from {phi_adapter_dir}")
        adapter_path = str(phi_adapter_dir)
        # Load Phi-3 base model with adapter
        model, tokenizer = load("mlx-community/Phi-3-mini-4k-instruct-4bit", 
                               adapter_path=adapter_path)
        print("✅ Fine-tuned Phi-3 model loaded successfully")
    else:
        print("No fine-tuned model found, loading base Llama model")
        # Load base Llama model without adapter
        model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
        print("✅ Base Llama model loaded successfully")

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
        if adapter_path and "llama3" in adapter_path:
            # Use the Llama format from training data
            formatted_prompt = f"<|user|>\n{request.prompt}<|end|>\n<|assistant|>\n"
        elif adapter_path:
            # Use Phi-3 format
            prompt_lower = request.prompt.strip().lower()
            if prompt_lower.endswith("looking for"):
                # Add guidance to avoid "to" immediately after "looking for"
                formatted_prompt = f"prompt: {request.prompt} someone completion:"
            else:
                formatted_prompt = f"prompt: {request.prompt} completion:"
        else:
            # For base model, use instruction format
            formatted_prompt = f"<|user|>\nComplete this text naturally: {request.prompt}<|end|>\n<|assistant|>\n"
        
        # Generate completion
        # MLX doesn't accept temperature/top_p directly, use sampler
        sampler = make_sampler(
            temp=request.temperature,
            top_p=request.top_p
        )
        
        # Generate with stop tokens support
        stop_tokens = request.stop if request.stop else [".", "!", "?", "\n", "<|end|>"]
        
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            max_tokens=request.max_tokens,
            sampler=sampler,
            verbose=False
        )
        
        # Apply stop tokens manually since MLX doesn't support them directly
        for stop_token in stop_tokens:
            if stop_token in response:
                # Keep the stop token if it's punctuation
                if stop_token in [".", "!", "?"]:
                    response = response.split(stop_token)[0] + stop_token
                else:
                    response = response.split(stop_token)[0]
                break
        
        # Clean up the response
        print(f"DEBUG: Raw model response: '{response}'", flush=True)
        completion = strip_prompt_from_response(request.prompt, response)
        print(f"DEBUG: After strip_prompt: '{completion}'", flush=True)
        
        # Additional cleanup for bio completions
        if completion:
            # Remove any trailing ellipsis (but not single periods)
            print(f"DEBUG: Before ellipsis removal: '{completion}'", flush=True)
            # Only remove if it ends with multiple dots
            if completion.endswith("..."):
                completion = completion[:-3].strip()
            elif completion.endswith("…"):
                completion = completion[:-1].strip()
            print(f"DEBUG: After ellipsis removal: '{completion}'", flush=True)
            
            # Fix capitalization issues in completion
            completion = fix_final_capitalization(completion, request.prompt)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        print(f"DEBUG: Final completion being returned: '{completion}'", flush=True)
        # Determine model name based on what was loaded
        if adapter_path:
            if "llama3" in adapter_path:
                model_name = "llama3.2-mlx-finetuned"
            else:
                model_name = "phi3-mlx-finetuned"
        else:
            model_name = "llama3.2-mlx-base"
        
        return AutocompleteResponse(
            completion=completion,
            elapsed_ms=elapsed_ms,
            model_name=model_name
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