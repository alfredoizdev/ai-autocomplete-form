#!/usr/bin/env python3
import httpx
import asyncio

async def test():
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            'http://localhost:8003/api/autocomplete/mlx',
            json={'prompt': 'Seeking a gentle well hung', 'max_tokens': 50, 'stop': ['.', '!', '?']}
        )
        data = resp.json()
        completion = data['completion']
        print(f"Raw completion: '{completion}'")
        print(f"Length: {len(completion.split())} words")
        print(f"Ends with punctuation: {any(completion.rstrip().endswith(p) for p in ['.', '!', '?'])}")
        
        # Test with specific prompt
        resp2 = await client.post(
            'http://localhost:8003/api/autocomplete/mlx',
            json={'prompt': 'Seeking a gentle well hung', 'max_tokens': 20}
        )
        data2 = resp2.json()
        print(f"\nTest 'Seeking a gentle well hung':")
        print(f"Completion: '{data2['completion']}'")
        
asyncio.run(test())