#!/usr/bin/env python3
"""
Convert the cleaned sentences into training data format.
Creates natural prompt-completion pairs from the sentences.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

def split_sentence_naturally(sentence: str) -> Tuple[str, str]:
    """
    Split a sentence into a natural prompt and completion.
    The split should feel like an incomplete thought that needs finishing.
    """
    words = sentence.split()
    total_words = len(words)
    
    # Aim for 40-60% of the sentence as prompt
    ideal_split = int(total_words * 0.4)
    
    # Find natural breaking points
    best_split = ideal_split
    best_score = 0
    
    # Words that often precede natural continuation points
    continuation_words = {
        'to': 3, 'with': 3, 'for': 3, 'and': 2, 'that': 3,
        'who': 4, 'which': 3, 'but': 2, 'or': 2, 'if': 2,
        'when': 3, 'where': 3, 'like': 2, 'about': 2, 'in': 2
    }
    
    # Check positions around the ideal split
    for i in range(max(3, ideal_split - 3), min(total_words - 3, ideal_split + 4)):
        if i < total_words - 1:
            word = words[i].lower().rstrip('.,!?;:')
            score = continuation_words.get(word, 0)
            
            # Bonus for certain patterns
            if i < total_words - 2:
                next_word = words[i + 1].lower()
                # "looking for" is a great split point
                if word == 'looking' and next_word == 'for':
                    score += 5
                # "interested in" is another good one
                elif word == 'interested' and next_word == 'in':
                    score += 4
                # "want to" / "like to" patterns
                elif word in ['want', 'like', 'need', 'love'] and next_word == 'to':
                    score += 4
            
            if score > best_score:
                best_score = score
                best_split = i + 1
    
    # If no good natural break, just use the ideal split
    if best_score == 0:
        best_split = ideal_split
    
    prompt = ' '.join(words[:best_split])
    completion = ' '.join(words[best_split:])
    
    return prompt, completion

def create_training_data(input_file: Path, output_dir: Path):
    """Create training data from cleaned sentences."""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the cleaned sentences
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        sentences_data = json.load(f)
    
    print(f"Found {len(sentences_data)} sentences")
    
    # Create prompt-completion pairs
    training_pairs = []
    
    for item in sentences_data:
        sentence = item['sentence']
        word_count = item['word_count']
        
        # Only use sentences with at least 10 words for good splits
        if word_count >= 10:
            prompt, completion = split_sentence_naturally(sentence)
            
            # Ensure both parts have substance
            if len(prompt.split()) >= 4 and len(completion.split()) >= 3:
                training_pairs.append({
                    'prompt': prompt,
                    'completion': completion,
                    'full_sentence': sentence,
                    'word_count': word_count
                })
    
    print(f"Created {len(training_pairs)} training pairs")
    
    # Shuffle the data
    random.shuffle(training_pairs)
    
    # Split into train/validation/test sets (80/10/10)
    total = len(training_pairs)
    train_size = int(total * 0.8)
    val_size = int(total * 0.1)
    
    train_data = training_pairs[:train_size]
    val_data = training_pairs[train_size:train_size + val_size]
    test_data = training_pairs[train_size + val_size:]
    
    print(f"\nDataset splits:")
    print(f"  Training: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    # Save in different formats
    
    # 1. JSON format for general use
    json_dir = output_dir / 'json'
    json_dir.mkdir(exist_ok=True)
    
    for name, data in [('train', train_data), ('validation', val_data), ('test', test_data)]:
        with open(json_dir / f'{name}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    # 2. JSONL format for fine-tuning
    jsonl_dir = output_dir / 'jsonl'
    jsonl_dir.mkdir(exist_ok=True)
    
    for name, data in [('train', train_data), ('validation', val_data), ('test', test_data)]:
        with open(jsonl_dir / f'{name}.jsonl', 'w', encoding='utf-8') as f:
            for item in data:
                # Format for OpenAI fine-tuning
                formatted = {
                    "messages": [
                        {"role": "user", "content": f"Complete this sentence: {item['prompt']}"},
                        {"role": "assistant", "content": item['completion']}
                    ]
                }
                f.write(json.dumps(formatted, ensure_ascii=False) + '\n')
    
    # 3. Simple text format for MLX
    mlx_dir = output_dir / 'mlx'
    mlx_dir.mkdir(exist_ok=True)
    
    for name, data in [('train', train_data), ('validation', val_data), ('test', test_data)]:
        with open(mlx_dir / f'{name}.jsonl', 'w', encoding='utf-8') as f:
            for item in data:
                # MLX format
                formatted = {
                    "text": f"<|user|>\n{item['prompt']}<|end|>\n<|assistant|>\n{item['completion']}<|end|>"
                }
                f.write(json.dumps(formatted, ensure_ascii=False) + '\n')
    
    # 4. Create sample file for review
    sample_file = output_dir / 'samples.txt'
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write("SAMPLE TRAINING PAIRS\n")
        f.write("="*60 + "\n\n")
        
        # Show diverse samples
        samples = random.sample(training_pairs, min(50, len(training_pairs)))
        
        for i, item in enumerate(samples, 1):
            f.write(f"Sample {i}:\n")
            f.write(f"  Prompt: {item['prompt']}\n")
            f.write(f"  Completion: {item['completion']}\n")
            f.write(f"  Full: {item['full_sentence']}\n")
            f.write("\n")
    
    print(f"\n✅ Training data created successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Formats created: JSON, JSONL, MLX")
    print(f"📋 Sample file: {sample_file}")
    
    # Print statistics
    print("\n" + "="*60)
    print("TRAINING DATA STATISTICS")
    print("="*60)
    
    prompt_lengths = [len(item['prompt'].split()) for item in training_pairs]
    completion_lengths = [len(item['completion'].split()) for item in training_pairs]
    
    print(f"\nPrompt statistics:")
    print(f"  Average length: {sum(prompt_lengths)/len(prompt_lengths):.1f} words")
    print(f"  Min length: {min(prompt_lengths)} words")
    print(f"  Max length: {max(prompt_lengths)} words")
    
    print(f"\nCompletion statistics:")
    print(f"  Average length: {sum(completion_lengths)/len(completion_lengths):.1f} words")
    print(f"  Min length: {min(completion_lengths)} words")
    print(f"  Max length: {max(completion_lengths)} words")
    
    # Show some examples of natural splits
    print("\n" + "="*60)
    print("EXAMPLES OF NATURAL SPLITS")
    print("="*60)
    
    natural_examples = [
        item for item in training_pairs[:100]
        if any(word in item['prompt'].lower() 
               for word in ['looking for', 'interested in', 'want to', 'like to'])
    ]
    
    for i, item in enumerate(natural_examples[:5], 1):
        print(f"\n{i}. '{item['prompt']}' → '{item['completion']}'")

def main():
    """Main function."""
    base_dir = Path(__file__).parent.parent.parent
    input_file = base_dir / 'data' / 'bio_sentences.json'
    output_dir = base_dir / 'data' / 'sentence_training'
    
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        print("Please run extract_quality_sentences.py first.")
        return
    
    create_training_data(input_file, output_dir)

if __name__ == "__main__":
    main()