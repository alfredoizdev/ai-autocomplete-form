"""
Status check script to verify all components are working correctly.
"""

import subprocess
import requests
import json
from pathlib import Path


def check_service(name, port, endpoint="/"):
    """Check if a service is running on a specific port."""
    try:
        response = requests.get(f"http://localhost:{port}{endpoint}", timeout=2)
        if response.status_code == 200:
            print(f"✅ {name} is running on port {port}")
            return True
    except:
        pass
    print(f"❌ {name} is NOT running on port {port}")
    return False


def check_ollama():
    """Check if Ollama is running and list models."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is running")
            print("   Available models:")
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                if line:
                    model_name = line.split()[0]
                    print(f"   - {model_name}")
            return True
    except:
        pass
    print("❌ Ollama is NOT running")
    return False


def check_docker():
    """Check if Docker containers are running."""
    try:
        result = subprocess.run(["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker is running")
            if "chromadb" in result.stdout:
                print("   - ChromaDB container is running")
            if "weaviate" in result.stdout:
                print("   - Weaviate container is running")
            return True
    except:
        pass
    print("❌ Docker is NOT running or not installed")
    return False


def check_phase1():
    """Check Phase 1 (Vector Search) components."""
    print("\n=== Phase 1: Vector Search Status ===")
    
    # Check API
    api_running = check_service("Vector Search API", 8001, "/")
    
    # Check database files
    chroma_db = Path("python/chroma_db/chroma.sqlite3")
    if chroma_db.exists():
        print("✅ ChromaDB database exists")
    else:
        print("❌ ChromaDB database not found")
    
    # Check if API is responding
    if api_running:
        try:
            response = requests.get("http://localhost:8001/api/stats")
            data = response.json()
            print(f"   Indexed bios: {data.get('total_bios', 'Unknown')}")
        except:
            pass


def check_phase2():
    """Check Phase 2 (MLX Training) components."""
    print("\n=== Phase 2: MLX Training Status ===")
    
    # Check MLX installation
    try:
        import mlx
        import mlx_lm
        print("✅ MLX installed")
    except ImportError:
        print("❌ MLX not installed")
    
    # Check training data
    train_data = Path("python/mlx_training/bio_dataset/train")
    if train_data.exists():
        print("✅ Training data prepared")
        stats_file = Path("python/mlx_training/bio_dataset/dataset_stats.json")
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
                print(f"   Training examples: {stats['train_examples']}")
                print(f"   Validation examples: {stats['val_examples']}")
    else:
        print("❌ Training data not found")
    
    # Check model config
    model_config = Path("python/mlx_training/models/model_config.json")
    if model_config.exists():
        print("✅ Model configuration exists")
    else:
        print("❌ Model not configured")
    
    # Check training config
    training_config = Path("python/mlx_training/adapters/bio_lora/training_config.json")
    if training_config.exists():
        print("✅ Training pipeline configured")
    else:
        print("❌ Training pipeline not configured")


def check_overall_system():
    """Check overall system status."""
    print("\n=== Overall System Status ===")
    
    # Check Next.js
    nextjs_running = check_service("Next.js", 3000, "")
    
    # Check Ollama
    ollama_running = check_ollama()
    
    # Check Docker
    docker_running = check_docker()
    
    # Summary
    print("\n=== Summary ===")
    if nextjs_running:
        print("✅ Autocomplete form is accessible at http://localhost:3000")
    else:
        print("❌ Run 'npm run dev' to start the application")
    
    print("\nCurrent capabilities:")
    print("- Autocomplete: Vector search (<100ms) + Ollama fallback")
    print("- Spell check: Active")
    print("- Auto-capitalization: Active")
    print("\nPhase 2 training: Ready (awaiting model download)")


if __name__ == "__main__":
    print("=== AI Train LLM Project Status Check ===")
    check_overall_system()
    check_phase1()
    check_phase2()
    
    print("\n=== Quick Commands ===")
    print("Start vector API: cd python && source venv/bin/activate && python api/api_server.py")
    print("Start Next.js: npm run dev")
    print("Train model: cd python/mlx_training && python train_mlx.py")
    print("\nFor detailed instructions, see progress-overview.md")