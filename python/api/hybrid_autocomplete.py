"""
Hybrid autocomplete: Combines vector search with trained model.
"""

import asyncio
from typing import List
import aiohttp

async def get_vector_suggestions(prompt: str) -> List[str]:
    """Get suggestions from vector search."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:8001/api/autocomplete',
                json={'prompt': prompt}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('suggestions', [])
    except:
        return []

async def get_trained_model_suggestions(prompt: str) -> List[str]:
    """Get suggestions from trained model."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:8002/api/autocomplete/trained',
                json={'prompt': prompt, 'max_suggestions': 2}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('suggestions', [])
    except:
        return []

async def get_hybrid_suggestions(prompt: str) -> List[str]:
    """Combine suggestions from both sources."""
    # Get suggestions from both sources in parallel
    vector_task = get_vector_suggestions(prompt)
    model_task = get_trained_model_suggestions(prompt)
    
    vector_suggestions, model_suggestions = await asyncio.gather(
        vector_task, model_task
    )
    
    # Combine and deduplicate
    all_suggestions = []
    
    # Add vector suggestions first (they're usually more accurate for exact matches)
    all_suggestions.extend(vector_suggestions[:2])
    
    # Add model suggestions
    for suggestion in model_suggestions:
        if suggestion not in all_suggestions:
            all_suggestions.append(suggestion)
    
    return all_suggestions[:3]  # Return top 3

if __name__ == "__main__":
    # Test the hybrid approach
    test_prompts = [
        "We are a couple who",
        "Looking for friends to",
        "My hobbies include"
    ]
    
    for prompt in test_prompts:
        suggestions = asyncio.run(get_hybrid_suggestions(prompt))
        print(f"\nPrompt: '{prompt}'")
        print(f"Suggestions: {suggestions}")
