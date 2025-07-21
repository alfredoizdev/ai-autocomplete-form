#!/usr/bin/env python3
"""
Convert bio training data to prompt-completion format where completions are single sentences only.
Creates multiple prompt-completion pairs from longer bios.
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

def load_bio_data(file_path: str) -> List[str]:
    """Load bio data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def fix_text_encoding(text):
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

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, handling common abbreviations."""
    # Protect common abbreviations
    text = re.sub(r'\b(Mr|Mrs|Ms|Dr|St|vs|etc|i\.e|e\.g)\.\s*', r'\1<DOT> ', text)
    
    # Split on sentence endings
    sentences = re.split(r'([.!?]+)\s+', text)
    
    # Reconstruct sentences with their punctuation
    result = []
    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            sent = sentences[i] + sentences[i + 1]
        else:
            sent = sentences[i]
        
        # Restore protected dots
        sent = sent.replace('<DOT>', '.')
        sent = sent.strip()
        
        if sent and len(sent.split()) >= 3:  # At least 3 words
            result.append(sent)
    
    return result

def find_split_point(sentence: str) -> Optional[Tuple[str, str]]:
    """Find a natural split point within a sentence for prompt-completion."""
    
    words = sentence.split()
    if len(words) < 6:  # Too short to split meaningfully
        return None
    
    # Strategy 1: Split at conjunctions and relative pronouns
    patterns = [
        (r'\b(who|that|which|where|when|whose|whom)\b', -1),
        (r'\b(because|since|although|though|while|whereas|if|unless|until|after|before)\b', -1),
        (r'\b(and|but|or|so|yet)\b', 0),
    ]
    
    for pattern, offset in patterns:
        matches = list(re.finditer(pattern, sentence, re.IGNORECASE))
        for match in matches:
            split_pos = match.end() if offset == 0 else match.start()
            
            # Check position is reasonable (20-70% through sentence)
            if 0.2 <= split_pos / len(sentence) <= 0.7:
                prompt = sentence[:split_pos].strip()
                completion = sentence[split_pos:].strip()
                
                prompt_words = len(prompt.split())
                completion_words = len(completion.split())
                
                if (3 <= prompt_words <= 40 and completion_words >= 3):
                    return prompt, completion
    
    # Strategy 2: Split at prepositions
    prep_patterns = [
        r'\b(looking for|searching for|interested in|seeking|hoping to find|want to meet)\b',
        r'\b(with|to|for|in|about|from)\b',
    ]
    
    for pattern in prep_patterns:
        matches = list(re.finditer(pattern, sentence, re.IGNORECASE))
        for match in matches:
            split_pos = match.end()
            
            # Add 1-2 words after the match
            remaining_text = sentence[split_pos:].strip()
            next_words = remaining_text.split()[:2]
            
            if next_words:
                extra_length = len(' '.join(next_words[:1]))  # Just 1 word after preposition
                split_pos += extra_length + 1
            
            if 0.15 <= split_pos / len(sentence) <= 0.7:
                prompt = sentence[:split_pos].strip()
                completion = sentence[split_pos:].strip()
                
                if (len(prompt.split()) >= 3 and len(completion.split()) >= 3):
                    return prompt, completion
    
    # Strategy 3: Split at commas (mid-sentence only)
    if ',' in sentence:
        comma_pos = sentence.find(',')
        if 0.2 <= comma_pos / len(sentence) <= 0.6:
            prompt = sentence[:comma_pos + 1].strip()  # Include comma
            completion = sentence[comma_pos + 1:].strip()
            
            if len(prompt.split()) >= 3 and len(completion.split()) >= 3:
                return prompt, completion
    
    # Strategy 4: Split at roughly 40% through sentence
    words = sentence.split()
    if len(words) >= 8:
        split_index = int(len(words) * 0.4)
        split_index = max(3, min(split_index, len(words) - 3))
        
        prompt = ' '.join(words[:split_index])
        completion = ' '.join(words[split_index:])
        
        if not prompt.endswith(('.', '!', '?', ':')):
            return prompt, completion
    
    return None

def create_prompt_completion_pairs(bio: str) -> List[Dict[str, str]]:
    """Create multiple prompt-completion pairs from a bio."""
    pairs = []
    
    # Clean the bio
    bio = fix_text_encoding(bio.strip())
    
    # Skip very short bios
    if len(bio.split()) <= 5:
        return []
    
    # Split into sentences
    sentences = split_into_sentences(bio)
    
    if not sentences:
        return []
    
    # Process each sentence
    remaining_text = ""
    
    for i, sentence in enumerate(sentences):
        # If we have remaining text from previous sentence, use it as prompt
        if remaining_text and len(remaining_text.split()) >= 3:
            # Complete the previous thought with start of current sentence
            words = sentence.split()
            if len(words) >= 4:
                # Take first few words to complete the thought
                completion_words = words[:min(len(words)//2, 10)]
                completion = ' '.join(completion_words)
                
                # Make sure completion ends properly
                if not completion.endswith(('.', '!', '?')):
                    # Find the end of the clause
                    for j, word in enumerate(completion_words):
                        if word.endswith(('.', '!', '?', ',', ';')):
                            completion = ' '.join(completion_words[:j+1])
                            break
                
                if len(completion.split()) >= 3:
                    pairs.append({
                        "prompt": remaining_text,
                        "completion": completion
                    })
                    
                    # Update remaining text
                    remaining_start = len(completion)
                    if remaining_start < len(sentence):
                        remaining_text = sentence[remaining_start:].strip()
                    else:
                        remaining_text = ""
                    continue
        
        # Try to split the current sentence
        split_result = find_split_point(sentence)
        
        if split_result:
            prompt, completion = split_result
            
            # If completion has multiple sentences, only take the first
            completion_sentences = split_into_sentences(completion)
            if completion_sentences:
                completion = completion_sentences[0]
                
                # Save remaining for next iteration
                if len(completion_sentences) > 1:
                    remaining_text = ' '.join(completion_sentences[1:])
                else:
                    remaining_text = ""
            
            pairs.append({
                "prompt": prompt,
                "completion": completion
            })
        else:
            # If we can't split it well, save for next iteration
            if remaining_text:
                remaining_text += " " + sentence
            else:
                remaining_text = sentence
    
    # Handle any final remaining text
    if remaining_text and len(remaining_text.split()) >= 6:
        split_result = find_split_point(remaining_text)
        if split_result:
            prompt, completion = split_result
            pairs.append({
                "prompt": prompt,
                "completion": completion
            })
    
    return pairs

def filter_by_total_length(prompt: str, completion: str, max_words: int = 512) -> bool:
    """Check if combined prompt + completion exceeds max words."""
    total_words = len(prompt.split()) + len(completion.split())
    return total_words <= max_words

def analyze_data(data: List[Dict[str, str]]):
    """Analyze the quality of prompt-completion pairs."""
    prompt_lengths = [len(entry['prompt'].split()) for entry in data]
    completion_lengths = [len(entry['completion'].split()) for entry in data]
    
    # Check for multi-sentence completions
    multi_sentence = 0
    for entry in data:
        if len(split_into_sentences(entry['completion'])) > 1:
            multi_sentence += 1
    
    print("\nData Analysis:")
    print(f"  Total pairs: {len(data)}")
    print(f"  Average prompt length: {sum(prompt_lengths)/len(prompt_lengths):.1f} words")
    print(f"  Average completion length: {sum(completion_lengths)/len(completion_lengths):.1f} words")
    print(f"  Min/Max prompt: {min(prompt_lengths)}/{max(prompt_lengths)} words")
    print(f"  Min/Max completion: {min(completion_lengths)}/{max(completion_lengths)} words")
    print(f"  Multi-sentence completions: {multi_sentence} ({multi_sentence/len(data)*100:.1f}%)")
    
    # Sample some data
    print("\nSample prompt-completion pairs:")
    samples = random.sample(data, min(5, len(data)))
    for i, entry in enumerate(samples, 1):
        print(f"\n  {i}. Prompt: '{entry['prompt']}'")
        print(f"     Completion: '{entry['completion']}'")

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

def save_jsonl(data: List[Dict[str, str]], file_path: str):
    """Save data in JSONL format."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def main():
    # Paths
    input_file = Path("../../data/bio.json")
    output_dir = Path("bio_mlx_single_sentence")
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading bio data...")
    bio_data = load_bio_data(input_file)
    print(f"Loaded {len(bio_data)} bios")
    
    # Convert to prompt-completion pairs
    print("\nCreating single-sentence prompt-completion pairs...")
    all_pairs = []
    
    for bio in bio_data:
        pairs = create_prompt_completion_pairs(bio)
        all_pairs.extend(pairs)
    
    # Filter by length
    filtered_pairs = []
    for pair in all_pairs:
        if filter_by_total_length(pair['prompt'], pair['completion']):
            filtered_pairs.append(pair)
    
    print(f"\nCreated {len(filtered_pairs)} prompt-completion pairs")
    print(f"(Generated from {len(bio_data)} original bios)")
    
    # Analyze the data
    analyze_data(filtered_pairs)
    
    # Split dataset
    print("\nSplitting dataset...")
    train_data, val_data, test_data = split_data(filtered_pairs)
    
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
    
    # Save examples
    examples_file = output_dir / "examples.txt"
    with open(examples_file, 'w', encoding='utf-8') as f:
        f.write("=== Single-Sentence Prompt-Completion Examples ===\n\n")
        
        examples = random.sample(train_data, min(30, len(train_data)))
        for i, entry in enumerate(examples):
            f.write(f"Example {i+1}:\n")
            f.write(f"PROMPT: {entry['prompt']}\n")
            f.write(f"COMPLETION: {entry['completion']}\n")
            f.write(f"FULL: {entry['prompt']} {entry['completion']}\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"Examples saved to {examples_file}")

if __name__ == "__main__":
    main()