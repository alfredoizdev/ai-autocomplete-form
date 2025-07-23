"""
Fast data preparation for MLX training with quality filtering.
Optimized for speed while maintaining quality.
"""

import json
import os
import csv
import re
from typing import List, Tuple, Optional
import random
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Fix encoding issues
    replacements = {
        'â€™': "'", 'â€œ': '"', 'â€': '"', 'â€"': '—',
        'â€"': '–', 'â€¦': '...', 'Ã©': 'é', 'Ã¨': 'è',
        'Ã ': 'à', 'Ã§': 'ç', 'Ã±': 'ñ', 'Ã¼': 'ü'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    text = text.strip()
    
    return text

def find_quality_splits_fast(text: str) -> List[Tuple[str, str, float]]:
    """Find high-quality split points using regex patterns."""
    splits = []
    
    # High-quality patterns for natural breaks
    patterns = [
        # After "looking for" phrases
        (r'(.*?(?:looking for|seeking|interested in|want to)\s+\S+)', 0.9),
        # After conjunctions with context
        (r'(.*?\b(?:and|but|so)\s+\S+\s+\S+)', 0.8),
        # After relative clauses
        (r'(.*?\b(?:who|which|that)\s+\S+\s+\S+)', 0.75),
        # After comma with substantial content
        (r'(.*?,\s*\S+\s+\S+)', 0.7),
    ]
    
    for pattern, quality in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        for match in matches:
            split_pos = match.end()
            
            # Skip if too close to start/end
            if split_pos < 25 or split_pos > len(text) - 20:
                continue
                
            prompt = text[:split_pos].strip()
            completion = text[split_pos:].strip()
            
            # Quick validation
            if validate_split_fast(prompt, completion):
                splits.append((prompt, completion, quality))
    
    # Sort by quality and return best splits
    splits.sort(key=lambda x: x[2], reverse=True)
    
    # Return top 3 splits to avoid processing too many
    return splits[:3]

def validate_split_fast(prompt: str, completion: str) -> bool:
    """Fast validation with essential checks only."""
    if not prompt or not completion:
        return False
    
    prompt_words = prompt.split()
    completion_words = completion.split()
    
    # Length requirements
    if len(prompt_words) < 5 or len(prompt_words) > 200:
        return False
    if len(completion_words) < 4 or len(completion_words) > 200:
        return False
    
    # Total must be substantial
    if len(prompt_words) + len(completion_words) < 12:
        return False
    
    # Prompt shouldn't end with sentence punctuation
    if re.search(r'[.!?]\s*$', prompt):
        return False
    
    # Completion must end with punctuation
    if not re.search(r'[.!?]\s*$', completion):
        return False
    
    # Quick grammar check for completion start
    first_word = completion.split()[0] if completion else ""
    if first_word and first_word[0].islower():
        allowed = {'and', 'but', 'or', 'who', 'which', 'that', 'where', 'when', 
                   'with', 'to', 'for', 'in', 'about', 'from'}
        if first_word.lower() not in allowed:
            return False
    
    return True

def process_bio_batch(bios: List[str]) -> List[dict]:
    """Process a batch of bios in parallel."""
    examples = []
    
    for bio in bios:
        # Split into sentences using simple regex
        sentences = re.split(r'(?<=[.!?])\s+', bio)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) < 10:
                continue
            
            # Get quality splits
            splits = find_quality_splits_fast(sentence)
            
            if splits:
                # Take best split
                prompt, completion, quality = splits[0]
                
                # Format for Llama model
                formatted_text = f"<|user|>\nComplete this bio: {prompt}<|end|>\n<|assistant|>\n{completion}<|end|>"
                
                examples.append({
                    'text': formatted_text,
                    'quality': quality
                })
    
    return examples

def process_dataset_fast(input_file: str, output_dir: str):
    """Process dataset with parallel processing for speed."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Read data
    print(f"Reading data from {input_file}...")
    bios = []
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                bio = clean_text(row[0])
                if len(bio.split()) >= 12:
                    bios.append(bio)
    
    print(f"Loaded {len(bios)} quality bios")
    
    # Process in batches with multiprocessing
    batch_size = 100
    batches = [bios[i:i+batch_size] for i in range(0, len(bios), batch_size)]
    
    print("Processing bios in parallel...")
    all_examples = []
    
    # Use simpler sequential processing for reliability
    for batch in tqdm(batches, desc="Processing batches"):
        batch_examples = process_bio_batch(batch)
        all_examples.extend(batch_examples)
    
    print(f"\nGenerated {len(all_examples)} training examples")
    
    # Filter for quality
    high_quality = [ex for ex in all_examples if ex['quality'] >= 0.7]
    print(f"High quality examples (>=0.7): {len(high_quality)}")
    
    # Use high quality examples
    all_examples = high_quality
    
    # Shuffle and split
    random.shuffle(all_examples)
    
    train_size = int(0.8 * len(all_examples))
    valid_size = int(0.1 * len(all_examples))
    
    train_data = all_examples[:train_size]
    valid_data = all_examples[train_size:train_size + valid_size]
    test_data = all_examples[train_size + valid_size:]
    
    # Save datasets
    for name, data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
        output_file = os.path.join(output_dir, f'{name}.jsonl')
        with open(output_file, 'w') as f:
            for item in data:
                # Only save the text field for training
                f.write(json.dumps({'text': item['text']}) + '\n')
        print(f"Saved {len(data)} examples to {output_file}")
    
    # Save quality statistics
    qualities = [ex['quality'] for ex in all_examples]
    print(f"\nQuality statistics:")
    print(f"  Average: {sum(qualities)/len(qualities):.3f}")
    print(f"  Min: {min(qualities):.3f}")
    print(f"  Max: {max(qualities):.3f}")

if __name__ == "__main__":
    # Process the LookingFor dataset with improved quality
    process_dataset_fast(
        "../../data/LookingFor_20000.csv",
        "lookingfor_hq"
    )