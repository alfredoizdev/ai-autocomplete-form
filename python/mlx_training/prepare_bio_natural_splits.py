#!/usr/bin/env python3
"""
Convert bio training data to prompt-completion format by naturally splitting the bio text.
This creates realistic training data where the prompt is the beginning of the bio
and the completion is the rest.
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple

def load_bio_data(file_path: str) -> List[str]:
    """Load bio data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_natural_split_point(bio: str) -> Tuple[str, str]:
    """Find a natural point to split the bio into prompt and completion."""
    
    # Clean up the bio
    bio = bio.strip()
    
    # Skip very short bios
    words = bio.split()
    if len(words) < 10:
        # For short bios, take first 2-3 words as prompt
        if len(words) >= 3:
            prompt = ' '.join(words[:2])
            completion = ' '.join(words[2:])
            return prompt, completion
        else:
            return bio, ""
    
    # Strategy 1: Split at punctuation marks (., !, ?)
    sentences = re.split(r'([.!?])', bio)
    if len(sentences) >= 3:  # At least 2 complete sentences
        # Take the first sentence as prompt
        first_sentence = sentences[0] + (sentences[1] if len(sentences) > 1 and sentences[1] in '.!?' else '')
        rest = ''.join(sentences[2:]) if len(sentences) > 2 else ''
        
        if first_sentence and rest and len(first_sentence.split()) >= 3 and len(rest.split()) >= 5:
            return first_sentence.strip(), rest.strip()
    
    # Strategy 2: Split at conjunctions or transitional phrases
    conjunctions = [
        ' and ', ' but ', ' so ', ' because ', ' since ', ' while ', ' although ',
        ' however ', ' therefore ', ' also ', ' we ', ' i ', ' our ', ' my '
    ]
    
    bio_lower = bio.lower()
    best_split = None
    best_score = 0
    
    for conj in conjunctions:
        if conj in bio_lower:
            # Find all occurrences
            start = 0
            while True:
                pos = bio_lower.find(conj, start)
                if pos == -1:
                    break
                
                # Check if this is a good split point (20-50% through the text)
                ratio = pos / len(bio)
                if 0.2 <= ratio <= 0.5:
                    prompt = bio[:pos].strip()
                    completion = bio[pos:].strip()
                    
                    # Score based on word counts
                    prompt_words = len(prompt.split())
                    completion_words = len(completion.split())
                    
                    if 5 <= prompt_words <= 30 and completion_words >= 10:
                        score = min(prompt_words, 30) + min(completion_words / 2, 50)
                        if score > best_score:
                            best_score = score
                            best_split = (prompt, completion)
                
                start = pos + 1
    
    if best_split:
        return best_split
    
    # Strategy 3: Split at commas for longer bios
    if ',' in bio and len(bio) > 100:
        parts = bio.split(',')
        if len(parts) >= 2:
            # Find a comma that's roughly 20-40% through the text
            cumulative_length = 0
            for i, part in enumerate(parts[:-1]):
                cumulative_length += len(part)
                ratio = cumulative_length / len(bio)
                
                if 0.2 <= ratio <= 0.4 and i > 0:
                    prompt = ','.join(parts[:i+1])
                    completion = ','.join(parts[i+1:])
                    
                    if len(prompt.split()) >= 5 and len(completion.split()) >= 10:
                        return prompt.strip(), completion.strip()
    
    # Strategy 4: Split by word count (fallback)
    words = bio.split()
    split_point = max(5, min(len(words) // 3, 25))  # Take 1/3 of words, but between 5-25 words
    
    prompt = ' '.join(words[:split_point])
    completion = ' '.join(words[split_point:])
    
    return prompt, completion

def filter_by_total_length(prompt: str, completion: str, max_words: int = 512) -> bool:
    """Check if combined prompt + completion exceeds max words."""
    total_words = len(prompt.split()) + len(completion.split())
    return total_words <= max_words and len(completion.strip()) > 0

def analyze_splits(data: List[Dict[str, str]]):
    """Analyze the quality of prompt-completion splits."""
    prompt_lengths = [len(entry['prompt'].split()) for entry in data]
    completion_lengths = [len(entry['completion'].split()) for entry in data]
    
    print("\nSplit Analysis:")
    print(f"  Average prompt length: {sum(prompt_lengths)/len(prompt_lengths):.1f} words")
    print(f"  Average completion length: {sum(completion_lengths)/len(completion_lengths):.1f} words")
    print(f"  Min/Max prompt: {min(prompt_lengths)}/{max(prompt_lengths)} words")
    print(f"  Min/Max completion: {min(completion_lengths)}/{max(completion_lengths)} words")
    
    # Show distribution
    print("\nPrompt length distribution:")
    ranges = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 100)]
    for low, high in ranges:
        count = sum(1 for l in prompt_lengths if low <= l < high)
        if count > 0:
            print(f"  {low}-{high} words: {count} ({count/len(prompt_lengths)*100:.1f}%)")

def split_data(data: List[Dict[str, str]], 
                train_ratio: float = 0.8, 
                val_ratio: float = 0.1) -> tuple:
    """Split data into train, validation, and test sets."""
    random.seed(42)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    total = len(shuffled_data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:train_size + val_size]
    test_data = shuffled_data[train_size + val_size:]
    
    return train_data, val_data, test_data

def fix_unicode_escapes(text: str) -> str:
    """Convert Unicode escape sequences to actual characters."""
    # Common replacements
    replacements = {
        '\\u2019': "'",  # Right single quotation mark
        '\\u2018': "'",  # Left single quotation mark
        '\\u201c': '"',  # Left double quotation mark
        '\\u201d': '"',  # Right double quotation mark
        '\\u2026': '...',  # Horizontal ellipsis
        '\\u2014': '—',  # Em dash
        '\\u2013': '–',  # En dash
        '\\u00a0': ' ',  # Non-breaking space
        '\\u00e9': 'é',  # e with acute accent
        '\\u00e8': 'è',  # e with grave accent
        '\\u00e0': 'à',  # a with grave accent
        '\\u00f1': 'ñ',  # n with tilde
        '\\u00fc': 'ü',  # u with umlaut
    }
    
    result = text
    for escaped, char in replacements.items():
        result = result.replace(escaped, char)
    
    # Try to decode any remaining unicode escapes
    try:
        result = result.encode().decode('unicode-escape')
    except:
        pass
    
    return result

def save_jsonl(data: List[Dict[str, str]], file_path: str):
    """Save data in JSONL format with proper Unicode handling."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            # Fix unicode escapes in both prompt and completion
            cleaned_entry = {
                "prompt": fix_unicode_escapes(entry["prompt"]),
                "completion": fix_unicode_escapes(entry["completion"])
            }
            # Use ensure_ascii=False to write actual Unicode characters
            f.write(json.dumps(cleaned_entry, ensure_ascii=False) + '\n')

def main():
    # Paths
    input_file = Path("../../data/bio.json")
    output_dir = Path("bio_mlx_natural")
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading bio data...")
    bio_data = load_bio_data(input_file)
    print(f"Loaded {len(bio_data)} bios")
    
    # Convert to prompt-completion pairs
    print("\nSplitting bios into natural prompt-completion pairs...")
    prompt_completion_data = []
    filtered_out = 0
    empty_completions = 0
    
    for bio in bio_data:
        prompt, completion = find_natural_split_point(bio)
        
        if not completion.strip():
            empty_completions += 1
            continue
            
        if filter_by_total_length(prompt, completion, max_words=512):
            prompt_completion_data.append({
                "prompt": prompt,
                "completion": completion
            })
        else:
            filtered_out += 1
    
    print(f"Created {len(prompt_completion_data)} prompt-completion pairs")
    print(f"Filtered out {filtered_out} entries exceeding 512 words")
    print(f"Skipped {empty_completions} entries with empty completions")
    
    # Analyze the splits
    analyze_splits(prompt_completion_data)
    
    # Split data
    print("\nSplitting data...")
    train_data, val_data, test_data = split_data(prompt_completion_data)
    
    print(f"\nData split:")
    print(f"  Train: {len(train_data)} entries")
    print(f"  Validation: {len(val_data)} entries")
    print(f"  Test: {len(test_data)} entries")
    
    # Save data
    print("\nSaving JSONL files...")
    save_jsonl(train_data, output_dir / "train.jsonl")
    save_jsonl(val_data, output_dir / "valid.jsonl")
    save_jsonl(test_data, output_dir / "test.jsonl")
    
    print(f"\nData saved to {output_dir}/")
    
    # Save some examples for review
    examples_file = output_dir / "examples.txt"
    with open(examples_file, 'w', encoding='utf-8') as f:
        f.write("=== Natural Prompt-Completion Splits ===\n\n")
        
        # Show some examples
        examples = random.sample(train_data, min(20, len(train_data)))
        for i, entry in enumerate(examples):
            # Fix unicode escapes for display
            prompt = fix_unicode_escapes(entry['prompt'])
            completion = fix_unicode_escapes(entry['completion'])
            f.write(f"Example {i+1}:\n")
            f.write(f"PROMPT: {prompt}\n")
            f.write(f"COMPLETION: {completion}\n")
            f.write(f"TOTAL WORDS: {len(prompt.split()) + len(completion.split())}\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"Sample examples saved to {examples_file}")
    
    # Show format example
    print("\nExample entries from train.jsonl:")
    for i in range(min(3, len(train_data))):
        print(f"\nEntry {i+1}:")
        print(f"  Prompt: {train_data[i]['prompt'][:80]}...")
        print(f"  Completion: {train_data[i]['completion'][:80]}...")
        print(f"  Lengths: {len(train_data[i]['prompt'].split())} + {len(train_data[i]['completion'].split())} words")

if __name__ == "__main__":
    main()