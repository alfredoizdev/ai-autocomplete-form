"""
Final evaluation script for bio autocomplete models.
Compares different models and provides recommendations.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
import time

# Model paths
models_to_test = {
    "DistilGPT2 (Failed)": Path(__file__).parent / "bio_distilgpt2_finetuned",
    "GPT2 Improved": Path(__file__).parent / "bio_gpt2_improved",
}

# Test prompts covering different bio scenarios
comprehensive_test_prompts = [
    # Couple introductions
    "We are a couple who",
    "Fun loving couple seeking",
    "Married couple looking for",
    "We are both professionals who",
    
    # Individual introductions
    "I am a single female",
    "I am looking for someone who",
    "Single male seeking",
    "I enjoy meeting new",
    
    # Activities and interests
    "Looking for friends to",
    "We enjoy outdoor activities like",
    "My hobbies include",
    "We love to travel and",
    
    # Specific interests
    "Seeking like-minded people for",
    "Interested in exploring",
    "We are new to",
    "Looking to expand our",
    
    # Preferences
    "We prefer couples who",
    "No single males",
    "Must be discreet and",
    "Age range",
]

def evaluate_model(model_name, model_path):
    """Evaluate a single model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Path: {model_path}")
    print('='*60)
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return None
    
    try:
        # Load model and tokenizer
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Move to device
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # Test generation parameters
        gen_params = {
            "max_new_tokens": 25,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.1
        }
        
        results = []
        total_time = 0
        
        print("\nGenerating completions...")
        for i, prompt in enumerate(comprehensive_test_prompts):
            start_time = time.time()
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_params)
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = generated[len(prompt):].strip()
            
            gen_time = time.time() - start_time
            total_time += gen_time
            
            results.append({
                "prompt": prompt,
                "completion": completion,
                "time": gen_time
            })
            
            # Print first 5 for quick review
            if i < 5:
                print(f"\n{i+1}. Prompt: '{prompt}'")
                print(f"   Completion: '{completion}'")
                print(f"   Time: {gen_time:.2f}s")
        
        avg_time = total_time / len(comprehensive_test_prompts)
        print(f"\nAverage generation time: {avg_time:.2f}s")
        
        return {
            "model_name": model_name,
            "results": results,
            "avg_time": avg_time,
            "total_prompts": len(results)
        }
        
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None

def analyze_results(all_results):
    """Analyze and compare results across models."""
    print("\n" + "="*60)
    print("ANALYSIS AND RECOMMENDATIONS")
    print("="*60)
    
    # Filter out failed models
    valid_results = [r for r in all_results if r is not None]
    
    if not valid_results:
        print("No models successfully evaluated!")
        return
    
    # Compare generation quality (manual inspection needed)
    print("\n1. Generation Quality:")
    print("   - GPT2 Improved: Shows bio-related vocabulary and structure")
    print("   - Needs manual review for coherence and appropriateness")
    
    # Compare generation speed
    print("\n2. Generation Speed:")
    for result in valid_results:
        print(f"   - {result['model_name']}: {result['avg_time']:.3f}s per completion")
    
    # Model size comparison
    print("\n3. Model Sizes:")
    for model_name, model_path in models_to_test.items():
        if model_path.exists():
            config_path = model_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    print(f"   - {model_name}: {config.get('model_type', 'Unknown')} architecture")
    
    print("\n4. Recommendations:")
    print("   a) For better quality:")
    print("      - Train GPT2 model on FULL dataset (35k examples)")
    print("      - Increase epochs to 3-4")
    print("      - Consider GPT2-medium (355M params) if memory allows")
    print("   ")
    print("   b) For production use:")
    print("      - Convert best model to ONNX for faster inference")
    print("      - Implement response caching for common prompts")
    print("      - Add content filtering for inappropriate completions")
    print("   ")
    print("   c) Next steps:")
    print("      1. Train on full dataset with current best settings")
    print("      2. Fine-tune generation parameters")
    print("      3. A/B test against current Gemma model")

def save_evaluation_report(all_results):
    """Save detailed evaluation report."""
    report_path = Path(__file__).parent / "evaluation_report.json"
    
    report = {
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_tested": len(all_results),
        "test_prompts": len(comprehensive_test_prompts),
        "results": all_results
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_path}")

def main():
    """Run comprehensive evaluation."""
    print("=== Bio Autocomplete Model Evaluation ===")
    print(f"Testing {len(models_to_test)} models with {len(comprehensive_test_prompts)} prompts")
    
    all_results = []
    
    for model_name, model_path in models_to_test.items():
        result = evaluate_model(model_name, model_path)
        all_results.append(result)
    
    # Analyze results
    analyze_results(all_results)
    
    # Save report
    save_evaluation_report([r for r in all_results if r is not None])
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()