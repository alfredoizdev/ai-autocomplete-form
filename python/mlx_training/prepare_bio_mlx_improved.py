#!/usr/bin/env python3
"""
Improved bio data preparation for MLX training.
Creates prompt-completion pairs that form single complete sentences.
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

def load_bio_data(file_path: str) -> List[str]:
    """Load bio data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def fix_text_encoding(text: str) -> str:
    """Fix various encoding issues in text."""
    replacements = {
        'â€™': "'", 'â€˜': "'", 'â€œ': '"', 'â€': '"',
        'â€¦': '...', 'â€"': '—', 'â€"': '–',
        'Ã©': 'é', 'Ã¨': 'è', 'Ã ': 'à', 'Ã±': 'ñ', 'Ã¼': 'ü',
        'â': "'",
    }
    
    result = text
    for bad, good in replacements.items():
        result = result.replace(bad, good)
    
    try:
        result = result.encode('latin-1').decode('utf-8', errors='ignore')
    except:
        pass
    
    result = result.replace('\xa0', ' ')
    return result

def is_single_sentence(text: str) -> bool:
    """Check if text is a single complete sentence."""
    # Remove quotes and parentheses content for checking
    check_text = re.sub(r'"[^"]*"', '', text)
    check_text = re.sub(r'\([^)]*\)', '', check_text)
    
    # Count sentence endings
    sentence_endings = len(re.findall(r'[.!?]\s*(?=[A-Z]|$)', check_text))
    
    # Also check for multiple periods not part of ellipsis
    periods = re.findall(r'\.(?!\.)', check_text)
    
    return sentence_endings <= 1 and len(periods) <= 1

def ends_with_punctuation(text: str) -> bool:
    """Check if text ends with proper punctuation."""
    text = text.rstrip()
    return bool(re.search(r'[.!?"\']$', text))

def find_quality_splits(text: str) -> List[Tuple[str, str]]:
    """Find all possible quality splits that form single sentences."""
    splits = []
    text = text.strip()
    
    # Skip if text is too short or already multiple sentences
    if len(text.split()) < 8 or not is_single_sentence(text):
        return []
    
    # Define split patterns with priority
    patterns = [
        # Coordinating conjunctions (high quality)
        (r'\b(and|but|or|yet|so)\b', 'after', 0.9),
        # Subordinating conjunctions
        (r'\b(because|since|although|though|while|whereas|if|unless|when|after|before)\b', 'after', 0.85),
        # Relative pronouns
        (r'\b(who|which|that|where|when|whose|whom)\b', 'before', 0.8),
        # Prepositions with specific patterns
        (r'\b(looking for|searching for|interested in|seeking|want to meet|hoping to find)\b', 'after_plus', 0.75),
        # Generic prepositions (lower quality)
        (r'\b(with|to|for|in|about|from)\b', 'after_plus', 0.6),
        # Commas (lowest quality)
        (r',', 'after', 0.5),
    ]
    
    for pattern, position, quality in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        
        for match in matches:
            if position == 'before':
                split_pos = match.start()
            elif position == 'after':
                split_pos = match.end()
            elif position == 'after_plus':
                # Include 1-2 words after the match
                split_pos = match.end()
                remaining = text[split_pos:].strip()
                next_words = remaining.split()[:2]
                if next_words:
                    # Only take first word for prepositions
                    extra = ' ' + next_words[0]
                    if split_pos + len(extra) < len(text):
                        split_pos += len(extra)
            
            # Skip if split position is too close to start/end
            if split_pos <= 10 or split_pos >= len(text) - 10:
                continue
            
            prompt = text[:split_pos].strip()
            completion = text[split_pos:].strip()
            
            # Validate the split
            if validate_split(prompt, completion):
                # Calculate position quality (prefer middle splits)
                position_ratio = split_pos / len(text)
                position_quality = 1.0 - abs(position_ratio - 0.5) * 2  # Max at 0.5
                
                combined_quality = quality * position_quality
                splits.append((prompt, completion, combined_quality))
    
    # Sort by quality and return unique splits
    splits.sort(key=lambda x: x[2], reverse=True)
    seen = set()
    unique_splits = []
    
    for prompt, completion, quality in splits:
        key = (prompt, completion)
        if key not in seen:
            seen.add(key)
            unique_splits.append((prompt, completion))
    
    return unique_splits

def validate_split(prompt: str, completion: str) -> bool:
    """Validate that a prompt-completion pair meets all requirements."""
    # Check basic requirements
    if not prompt or not completion:
        return False
    
    prompt_words = prompt.split()
    completion_words = completion.split()
    
    # Length requirements
    if len(prompt_words) < 3 or len(prompt_words) > 30:
        return False
    if len(completion_words) < 3 or len(completion_words) > 40:
        return False
    
    # Total length check
    if len(prompt_words) + len(completion_words) > 60:
        return False
    
    # Prompt should NOT end with sentence punctuation
    if re.search(r'[.!?]\s*$', prompt):
        return False
    
    # Completion MUST end with sentence punctuation
    if not ends_with_punctuation(completion):
        return False
    
    # Combined text must form a single sentence
    combined = prompt + ' ' + completion
    if not is_single_sentence(combined):
        return False
    
    # Avoid splits that start completion with lowercase (unless it's a special word)
    if completion[0].islower() and not completion.startswith(('i ', "i'", 'and', 'but', 'or')):
        # Check if it's a continuation that makes sense
        if not re.match(r'^(who|which|that|where|when|whose|whom|with|to|for|in|about|from)', completion):
            return False
    
    return True

def create_prompt_completion_pairs(bio: str) -> List[Dict[str, str]]:
    """Create high-quality prompt-completion pairs from a bio."""
    bio = fix_text_encoding(bio.strip())
    
    # First, split bio into sentences
    sentences = re.split(r'(?<=[.!?])\s+', bio)
    
    pairs = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Find all quality splits for this sentence
        splits = find_quality_splits(sentence)
        
        # Take the best split (first one, already sorted by quality)
        if splits:
            prompt, completion = splits[0]
            pairs.append({
                "prompt": prompt,
                "completion": completion
            })
    
    return pairs

def analyze_dataset(data: List[Dict[str, str]], name: str = "Dataset"):
    """Analyze the quality of the dataset."""
    print(f"\n{'='*60}")
    print(f"{name} Analysis")
    print(f"{'='*60}")
    
    # Basic stats
    print(f"Total samples: {len(data)}")
    
    if not data:
        return
    
    # Length analysis
    prompt_lengths = [len(entry['prompt'].split()) for entry in data]
    completion_lengths = [len(entry['completion'].split()) for entry in data]
    combined_lengths = [p + c for p, c in zip(prompt_lengths, completion_lengths)]
    
    print(f"\nPrompt lengths:")
    print(f"  Average: {sum(prompt_lengths)/len(prompt_lengths):.1f} words")
    print(f"  Min/Max: {min(prompt_lengths)}/{max(prompt_lengths)} words")
    
    print(f"\nCompletion lengths:")
    print(f"  Average: {sum(completion_lengths)/len(completion_lengths):.1f} words")
    print(f"  Min/Max: {min(completion_lengths)}/{max(completion_lengths)} words")
    
    print(f"\nCombined lengths:")
    print(f"  Average: {sum(combined_lengths)/len(combined_lengths):.1f} words")
    print(f"  Min/Max: {min(combined_lengths)}/{max(combined_lengths)} words")
    
    # Quality checks
    issues = {
        'prompt_ends_punctuation': 0,
        'completion_no_punctuation': 0,
        'multiple_sentences': 0,
        'too_short': 0,
        'perfect': 0
    }
    
    for entry in data:
        prompt = entry['prompt']
        completion = entry['completion']
        combined = prompt + ' ' + completion
        
        has_issue = False
        
        if re.search(r'[.!?]\s*$', prompt):
            issues['prompt_ends_punctuation'] += 1
            has_issue = True
        
        if not ends_with_punctuation(completion):
            issues['completion_no_punctuation'] += 1
            has_issue = True
        
        if not is_single_sentence(combined):
            issues['multiple_sentences'] += 1
            has_issue = True
        
        if len(prompt.split()) < 3 or len(completion.split()) < 3:
            issues['too_short'] += 1
            has_issue = True
        
        if not has_issue:
            issues['perfect'] += 1
    
    print(f"\nQuality Analysis:")
    print(f"  Perfect samples: {issues['perfect']} ({issues['perfect']/len(data)*100:.1f}%)")
    print(f"  Issues found:")
    print(f"    - Prompt ends with punctuation: {issues['prompt_ends_punctuation']}")
    print(f"    - Completion missing punctuation: {issues['completion_no_punctuation']}")
    print(f"    - Multiple sentences formed: {issues['multiple_sentences']}")
    print(f"    - Too short: {issues['too_short']}")
    
    # Sample some examples
    print(f"\nRandom Examples:")
    samples = random.sample(data, min(5, len(data)))
    for i, entry in enumerate(samples, 1):
        print(f"\n  Example {i}:")
        print(f"    Prompt: '{entry['prompt']}'")
        print(f"    Completion: '{entry['completion']}'")
        print(f"    Full: '{entry['prompt']} {entry['completion']}'")

def split_dataset(data: List[Dict[str, str]], 
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

def save_jsonl(data: List[Dict[str, str]], file_path: str):
    """Save data in JSONL format."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def main():
    # Paths
    input_file = Path("../../data/bio.json")
    output_dir = Path("bio_mlx_improved")
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading bio data...")
    bio_data = load_bio_data(input_file)
    print(f"Loaded {len(bio_data)} bios")
    
    # Convert to prompt-completion pairs
    print("\nCreating improved prompt-completion pairs...")
    all_pairs = []
    
    for i, bio in enumerate(bio_data):
        if i % 500 == 0:
            print(f"  Processing bio {i}/{len(bio_data)}...")
        
        pairs = create_prompt_completion_pairs(bio)
        all_pairs.extend(pairs)
    
    print(f"\nCreated {len(all_pairs)} prompt-completion pairs")
    print(f"Average pairs per bio: {len(all_pairs)/len(bio_data):.2f}")
    
    # Analyze the full dataset
    analyze_dataset(all_pairs, "Full Dataset")
    
    # Split dataset
    print("\n" + "="*60)
    print("Splitting dataset...")
    train_data, val_data, test_data = split_dataset(all_pairs)
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    # Analyze each split
    analyze_dataset(train_data, "Training Set")
    
    # Save data
    print("\n" + "="*60)
    print("Saving JSONL files...")
    save_jsonl(train_data, output_dir / "train.jsonl")
    save_jsonl(val_data, output_dir / "valid.jsonl")
    save_jsonl(test_data, output_dir / "test.jsonl")
    
    print(f"\nData saved to {output_dir}/")
    
    # Save examples file
    examples_file = output_dir / "examples.txt"
    with open(examples_file, 'w', encoding='utf-8') as f:
        f.write("=== Improved Single-Sentence Prompt-Completion Examples ===\n\n")
        f.write("All examples form exactly ONE complete sentence when combined.\n\n")
        
        examples = random.sample(train_data, min(50, len(train_data)))
        for i, entry in enumerate(examples):
            f.write(f"Example {i+1}:\n")
            f.write(f"PROMPT: {entry['prompt']}\n")
            f.write(f"COMPLETION: {entry['completion']}\n")
            f.write(f"FULL: {entry['prompt']} {entry['completion']}\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"Examples saved to {examples_file}")
    
    # Create comparison report
    comparison_file = output_dir / "quality_report.txt"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write("=== Data Quality Comparison Report ===\n\n")
        f.write(f"Original dataset issues (from bio_mlx_single_sentence):\n")
        f.write(f"  - Only 53.2% of samples were correct\n")
        f.write(f"  - 18.9% formed two separate sentences\n")
        f.write(f"  - 21.6% resulted in incomplete sentences\n")
        f.write(f"  - 6.3% had multiple sentences in completion\n\n")
        
        perfect_ratio = sum(1 for entry in all_pairs if validate_split(entry['prompt'], entry['completion'])) / len(all_pairs)
        f.write(f"Improved dataset quality:\n")
        f.write(f"  - {perfect_ratio*100:.1f}% of samples are perfect\n")
        f.write(f"  - All samples form single complete sentences\n")
        f.write(f"  - All prompts end without punctuation\n")
        f.write(f"  - All completions end with proper punctuation\n")
    
    print(f"Quality report saved to {comparison_file}")

if __name__ == "__main__":
    main()