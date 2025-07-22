#!/usr/bin/env python3
"""
Benchmark script for MLX model performance.
Tests response times and throughput.
"""

import httpx
import asyncio
import time
import statistics

async def benchmark_single_requests(client, prompts, runs=5):
    """Benchmark single request performance."""
    times = []
    
    print(f"\n📊 Single Request Benchmark ({runs} runs per prompt)")
    print("-" * 60)
    
    for prompt in prompts:
        prompt_times = []
        
        for _ in range(runs):
            start = time.time()
            response = await client.post(
                "http://localhost:8003/api/autocomplete/mlx",
                json={
                    "prompt": prompt,
                    "max_tokens": 30,
                    "temperature": 0.7
                }
            )
            elapsed = (time.time() - start) * 1000  # ms
            
            if response.status_code == 200:
                prompt_times.append(elapsed)
        
        if prompt_times:
            avg_time = statistics.mean(prompt_times)
            min_time = min(prompt_times)
            max_time = max(prompt_times)
            times.extend(prompt_times)
            
            print(f"Prompt: '{prompt[:30]}...'" if len(prompt) > 30 else f"Prompt: '{prompt}'")
            print(f"  Avg: {avg_time:.1f}ms | Min: {min_time:.1f}ms | Max: {max_time:.1f}ms")
    
    if times:
        print("\n📈 Overall Statistics:")
        print(f"  Mean: {statistics.mean(times):.1f}ms")
        print(f"  Median: {statistics.median(times):.1f}ms")
        print(f"  Std Dev: {statistics.stdev(times):.1f}ms")
        print(f"  95th percentile: {sorted(times)[int(len(times) * 0.95)]:.1f}ms")

async def benchmark_concurrent_requests(client, prompts, concurrent=5):
    """Benchmark concurrent request handling."""
    print(f"\n🔄 Concurrent Request Benchmark ({concurrent} simultaneous requests)")
    print("-" * 60)
    
    start = time.time()
    
    # Create concurrent tasks
    tasks = []
    for i in range(concurrent):
        prompt = prompts[i % len(prompts)]
        task = client.post(
            "http://localhost:8003/api/autocomplete/mlx",
            json={
                "prompt": prompt,
                "max_tokens": 30,
                "temperature": 0.7
            }
        )
        tasks.append(task)
    
    # Wait for all to complete
    responses = await asyncio.gather(*tasks)
    
    total_time = time.time() - start
    successful = sum(1 for r in responses if r.status_code == 200)
    
    print(f"Total time: {total_time:.2f}s")
    print(f"Successful: {successful}/{concurrent}")
    print(f"Throughput: {successful/total_time:.1f} requests/second")

async def benchmark_batch_endpoint(client, prompts):
    """Benchmark batch endpoint performance."""
    print(f"\n📦 Batch Endpoint Benchmark")
    print("-" * 60)
    
    batch_sizes = [3, 5]
    
    for size in batch_sizes:
        batch_prompts = prompts[:size]
        
        start = time.time()
        response = await client.post(
            "http://localhost:8003/api/autocomplete/mlx/batch",
            json=batch_prompts,
            timeout=30.0
        )
        elapsed = (time.time() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            print(f"Batch size {size}: {elapsed:.1f}ms total ({elapsed/size:.1f}ms per prompt)")
        else:
            print(f"Batch size {size}: Failed with status {response.status_code}")

async def main():
    """Run all benchmarks."""
    
    test_prompts = [
        "We are a couple looking for",
        "I enjoy meeting new people and",
        "My ideal evening involves",
        "Looking for friends who enjoy",
        "We love to explore",
        "Open-minded and looking for",
        "New to the lifestyle and",
        "Seeking like-minded people who"
    ]
    
    print("🚀 MLX Model Performance Benchmark")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Check if server is running
        try:
            response = await client.get("http://localhost:8003/")
            server_info = response.json()
            print(f"Server: {server_info['service']}")
            print(f"Model loaded: {server_info['model_loaded']}")
            print(f"Adapter loaded: {server_info['adapter_loaded']}")
        except Exception as e:
            print(f"❌ Server not running: {e}")
            return
        
        # Run benchmarks
        await benchmark_single_requests(client, test_prompts[:5])
        await benchmark_concurrent_requests(client, test_prompts)
        await benchmark_batch_endpoint(client, test_prompts)
        
        print("\n✅ Benchmark complete!")

if __name__ == "__main__":
    asyncio.run(main())