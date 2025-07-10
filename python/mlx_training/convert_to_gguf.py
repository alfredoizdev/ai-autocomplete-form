"""
Convert MLX fine-tuned model to GGUF format for Ollama.
This allows us to use the fine-tuned model with the existing Ollama infrastructure.
"""

import json
import subprocess
from pathlib import Path


def create_ollama_modelfile(adapter_path: Path, base_model: str = "gemma2:2b"):
    """
    Create a Modelfile for Ollama that includes our fine-tuned adapters.
    """
    modelfile_content = f"""# Bio Autocomplete Model
# Based on {base_model} with custom LoRA adapters

FROM {base_model}

# Model parameters optimized for bio completion
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER stop "\\n"
PARAMETER stop "."
PARAMETER stop "!"
PARAMETER stop "?"

# System prompt for bio completion
SYSTEM You are a bio completion assistant. Complete partial bios in a natural, engaging way. Keep completions concise (3-5 words) and contextually appropriate. Match the tone and style of the input.

# Template for bio completion
TEMPLATE {{{{ .Prompt }}}}
"""
    
    modelfile_path = adapter_path / "Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"Created Modelfile at: {modelfile_path}")
    return modelfile_path


def convert_mlx_to_gguf(adapter_path: Path):
    """
    Convert MLX adapters to GGUF format.
    Note: This is a simplified version. Full conversion requires the actual trained model.
    """
    print("=== MLX to GGUF Conversion ===")
    
    # Check if adapter path exists
    if not adapter_path.exists():
        print(f"❌ Adapter path not found: {adapter_path}")
        return None
    
    # Load training config
    config_path = adapter_path / "training_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config for model: {config['model_name']}")
    
    # Create conversion metadata
    conversion_meta = {
        "source": "mlx_training",
        "base_model": "gemma2:2b",
        "adapter_type": "lora",
        "training_examples": 4599,
        "status": "ready_for_conversion",
        "note": "Conversion requires trained model weights"
    }
    
    meta_path = adapter_path / "conversion_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(conversion_meta, f, indent=2)
    
    print(f"\nConversion metadata saved to: {meta_path}")
    
    # Create Ollama Modelfile
    modelfile_path = create_ollama_modelfile(adapter_path)
    
    return modelfile_path


def create_ollama_model(modelfile_path: Path, model_name: str = "bio-autocomplete"):
    """
    Create an Ollama model from the Modelfile.
    """
    print(f"\n=== Creating Ollama Model: {model_name} ===")
    
    # Command to create Ollama model
    cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nNote: This will create a model based on the existing gemma2:2b")
    print("Once you have trained adapters, they can be integrated into the model")
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"\n✅ Model '{model_name}' created successfully!")
            print("Output:", result.stdout)
        else:
            print(f"\n❌ Error creating model:")
            print("Error:", result.stderr)
            
    except FileNotFoundError:
        print("\n❌ Ollama command not found. Make sure Ollama is installed and running.")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def test_ollama_model(model_name: str = "bio-autocomplete"):
    """
    Test the created Ollama model.
    """
    print(f"\n=== Testing Model: {model_name} ===")
    
    test_prompts = [
        "We are a fun couple who",
        "Looking for friends to",
        "My hobbies include"
    ]
    
    for prompt in test_prompts:
        cmd = ["ollama", "run", model_name, prompt]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print(f"\nPrompt: '{prompt}'")
                print(f"Completion: '{result.stdout.strip()}'")
            else:
                print(f"\nError with prompt '{prompt}': {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"\nTimeout for prompt: '{prompt}'")
        except Exception as e:
            print(f"\nError testing prompt '{prompt}': {e}")


if __name__ == "__main__":
    # Path to adapters
    adapter_path = Path(__file__).parent / "adapters" / "bio_lora"
    
    # Convert to GGUF format
    modelfile_path = convert_mlx_to_gguf(adapter_path)
    
    if modelfile_path:
        print("\n" + "="*50)
        print("Ready to create Ollama model!")
        print("\nNext steps:")
        print("1. Train the model using train_mlx.py")
        print("2. Run this script again to convert trained weights")
        print("3. Create Ollama model with: ollama create bio-autocomplete -f", modelfile_path)
        print("4. Test with: ollama run bio-autocomplete 'We are a couple who'")
        
        # Optionally create the model now (using base model for testing)
        user_input = input("\nCreate test model now? (y/n): ")
        if user_input.lower() == 'y':
            create_ollama_model(modelfile_path, "bio-autocomplete-test")
            test_ollama_model("bio-autocomplete-test")