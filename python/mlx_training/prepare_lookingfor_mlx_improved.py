"""
Improved data preparation for MLX training with grammar quality filtering.
Creates higher quality prompt-completion pairs for grammatically correct outputs.
"""

import json
import os
import csv
import re
from typing import List, Tuple, Optional
import random
from collections import defaultdict
import spacy
from tqdm import tqdm

# Load spaCy model for better sentence parsing
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Installing spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

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

def get_natural_split_points(doc) -> List[Tuple[int, float]]:
    """
    Find natural split points using spaCy's dependency parsing.
    Returns list of (char_position, quality_score).
    """
    split_points = []
    text = doc.text
    
    for token in doc:
        # High quality splits: after clauses
        if token.dep_ in ['advcl', 'relcl', 'ccomp']:
            # Split after the clause
            clause_end = token.subtree
            last_token = list(clause_end)[-1]
            if last_token.i < len(doc) - 1:
                pos = last_token.idx + len(last_token.text)
                split_points.append((pos, 0.9))
        
        # Good splits: after conjunctions with proper context
        elif token.text.lower() in ['and', 'but', 'or', 'so'] and token.dep_ == 'cc':
            if token.i > 3 and token.i < len(doc) - 3:  # Ensure context
                pos = token.idx + len(token.text)
                split_points.append((pos, 0.8))
        
        # Medium splits: after certain punctuation
        elif token.text == ',' and token.i > 5 and token.i < len(doc) - 5:
            # Only split at commas that separate clauses
            if any(child.dep_ in ['nsubj', 'nsubjpass'] for child in token.nbor(1).subtree):
                pos = token.idx + 1
                split_points.append((pos, 0.7))
    
    # Add splits at "looking for" and similar phrases
    looking_patterns = [
        (r'\b(looking for|seeking|interested in|want to)\s+', 0.85),
        (r'\b(who|that|which)\s+', 0.75),
    ]
    
    for pattern, quality in looking_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            split_points.append((match.end(), quality))
    
    return split_points

def create_quality_split(text: str, min_prompt_words: int = 5) -> List[Tuple[str, str, float]]:
    """
    Create high-quality prompt-completion splits using NLP analysis.
    """
    doc = nlp(text)
    splits = []
    
    # Get natural split points
    split_points = get_natural_split_points(doc)
    
    for pos, quality in split_points:
        # Skip if too close to start/end
        if pos < 30 or pos > len(text) - 20:
            continue
            
        prompt = text[:pos].strip()
        completion = text[pos:].strip()
        
        # Enhanced validation
        if validate_split_enhanced(prompt, completion):
            splits.append((prompt, completion, quality))
    
    # Sort by quality
    splits.sort(key=lambda x: x[2], reverse=True)
    
    # Return only high-quality splits
    return [(p, c, q) for p, c, q in splits if q >= 0.7]

def validate_split_enhanced(prompt: str, completion: str) -> bool:
    """Enhanced validation with stricter requirements."""
    if not prompt or not completion:
        return False
    
    prompt_words = prompt.split()
    completion_words = completion.split()
    
    # Stricter length requirements
    if len(prompt_words) < 5 or len(prompt_words) > 200:
        return False
    if len(completion_words) < 4 or len(completion_words) > 200:
        return False
    
    # Total length check
    if len(prompt_words) + len(completion_words) > 400:
        return False
    
    # Combined must be substantial
    if len(prompt_words) + len(completion_words) < 12:
        return False
    
    # Prompt should end naturally (not mid-word, not with period)
    if re.search(r'[.!?]\s*$', prompt):
        return False
    if re.search(r'\w-$', prompt):  # Ends with hyphen
        return False
    
    # Completion must end with proper punctuation
    if not re.search(r'[.!?]\s*$', completion):
        return False
    
    # Check completion starts appropriately
    first_word = completion.split()[0] if completion else ""
    last_prompt_word = prompt_words[-1].lower().rstrip(',.:;')
    
    # If prompt ends with preposition, completion shouldn't start with verb
    if last_prompt_word in ['for', 'with', 'to', 'in', 'at', 'on', 'by', 'from', 'about']:
        if first_word.lower() in ['is', 'are', 'was', 'were', 'have', 'has', 'had']:
            return False
    
    # If completion starts lowercase, it should be a continuation word
    if first_word and first_word[0].islower():
        allowed_continuations = {
            'and', 'but', 'or', 'who', 'which', 'that', 'where', 'when',
            'whose', 'whom', 'with', 'to', 'for', 'in', 'about', 'from'
        }
        if first_word.lower() not in allowed_continuations:
            return False
    
    return True

def process_dataset(input_file: str, output_dir: str):
    """Process the dataset with improved quality filtering."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Read data
    print(f"Reading data from {input_file}...")
    bios = []
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                bio = clean_text(row[0])
                if len(bio.split()) >= 12:  # Minimum quality threshold
                    bios.append(bio)
    
    print(f"Loaded {len(bios)} quality bios")
    
    # Process into high-quality training pairs
    all_examples = []
    failed_count = 0
    
    for bio in tqdm(bios, desc="Processing bios"):
        # Split into sentences
        doc = nlp(bio)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        for sentence in sentences:
            if len(sentence.split()) < 10:
                continue
                
            # Get quality splits
            splits = create_quality_split(sentence)
            
            if splits:
                # Take best split
                prompt, completion, quality = splits[0]
                
                # Format for Llama model
                formatted_text = f"<|user|>\nComplete this bio: {prompt}<|end|>\n<|assistant|>\n{completion}<|end|>"
                
                all_examples.append({
                    'text': formatted_text,
                    'quality': quality,
                    'prompt_length': len(prompt.split()),
                    'completion_length': len(completion.split())
                })
            else:
                failed_count += 1
    
    print(f"\nGenerated {len(all_examples)} training examples")
    print(f"Failed to process: {failed_count} sentences")
    
    # Quality statistics
    qualities = [ex['quality'] for ex in all_examples]
    print(f"\nQuality distribution:")
    print(f"  Average: {sum(qualities)/len(qualities):.3f}")
    print(f"  Min: {min(qualities):.3f}")
    print(f"  Max: {max(qualities):.3f}")
    
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

if __name__ == "__main__":
    # Process the LookingFor dataset with improved quality
    process_dataset(
        "../../data/LookingFor_20000.csv",
        "lookingfor_hq"
    )