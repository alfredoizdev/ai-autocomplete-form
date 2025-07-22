#!/usr/bin/env python3
"""
Test script for MLX model server.
"""

import httpx
import asyncio
import time

async def test_server():
    """Test the MLX server endpoints."""
    
    base_url = "http://localhost:8003"
    
    # Test prompts
    test_prompts = [
        "We are a couple looking for",
        "I enjoy meeting new people and",
        "My ideal evening involves",
        "Looking for friends who enjoy",
        "We love to explore"
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Check health
        print("Testing health endpoint...")
        try:
            response = await client.get(f"{base_url}/")
            print(f"Health check: {response.json()}")
            print()
        except Exception as e:
            print(f"❌ Server not running: {e}")
            print("Please start the server with: python mlx_model_server.py")
            return
        
        # Test individual completions
        print("Testing individual completions:")
        print("-" * 50)
        
        for prompt in test_prompts:
            try:
                start = time.time()
                response = await client.post(
                    f"{base_url}/api/autocomplete/mlx",
                    json={
                        "prompt": prompt,
                        "max_tokens": 30,
                        "temperature": 0.7
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    elapsed = time.time() - start
                    
                    print(f"Prompt: {prompt}")
                    print(f"Completion: {data['completion']}")
                    print(f"Model: {data['model_name']}")
                    print(f"Time: {data['elapsed_ms']:.1f}ms (total: {elapsed:.2f}s)")
                    print("-" * 50)
                else:
                    print(f"Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"Error with prompt '{prompt}': {e}")
        
        # Test batch endpoint
        print("\nTesting batch completions:")
        print("-" * 50)
        
        try:
            response = await client.post(
                f"{base_url}/api/autocomplete/mlx/batch",
                json=test_prompts[:3]
            )
            
            if response.status_code == 200:
                data = response.json()
                for result in data['results']:
                    print(f"Prompt: {result['prompt']}")
                    print(f"Completion: {result['completion']}")
                    print()
            else:
                print(f"Batch error: {response.status_code}")
                
        except Exception as e:
            print(f"Batch error: {e}")

if __name__ == "__main__":
    print("🧪 MLX Server Test Suite")
    print("=" * 50)
    asyncio.run(test_server())