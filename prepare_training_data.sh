#!/bin/bash

# Prepare bio.json data for MLX training
# Converts raw bios into prompt-completion format

echo "📊 Preparing Training Data for MLX"
echo "=================================="
echo ""

# Configuration
SOURCE_FILE="data/bio.json"
OUTPUT_DIR="python/mlx_training/bio_mlx_improved"
TRAIN_SPLIT=0.8
VAL_SPLIT=0.1
TEST_SPLIT=0.1

echo "Configuration:"
echo "  Source: $SOURCE_FILE"
echo "  Output: $OUTPUT_DIR"
echo "  Train/Val/Test split: $TRAIN_SPLIT/$VAL_SPLIT/$TEST_SPLIT"
echo ""

# Check if source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "❌ Source file not found: $SOURCE_FILE"
    exit 1
fi

# Check if Python venv exists
if [ ! -d "python/venv" ]; then
    echo "❌ Python virtual environment not found!"
    echo "Please run: cd python && python -m venv venv && pip install -r requirements.txt"
    exit 1
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Create the data preparation script
cat > python/mlx_training/prepare_bio_mlx_improved.py << 'EOF'
#!/usr/bin/env python3
"""
Convert bio.json to MLX training format with intelligent prompt-completion splits
"""

import json
import random
from pathlib import Path
import re

def create_natural_split(bio_text):
    """
    Split bio text into a natural prompt and completion.
    Creates incomplete prompts that naturally lead to completions.
    """
    bio_text = bio_text.strip()
    
    # Skip very short bios
    if len(bio_text.split()) < 10:
        return None
        
    # Patterns for good split points (incomplete thoughts)
    split_patterns = [
        # Split after "looking for" + 0-2 words
        (r'(.*\blooking for\s+\w+(?:\s+\w+)?)\s+(.+)', 'partial'),
        # Split after "interested in" + 0-2 words  
        (r'(.*\binterested in\s+\w+(?:\s+\w+)?)\s+(.+)', 'partial'),
        # Split after "seeking" + 0-2 words
        (r'(.*\bseeking\s+\w+(?:\s+\w+)?)\s+(.+)', 'partial'),
        # Split after "we are" + 1-2 words
        (r'(.*\bwe are\s+\w+(?:\s+\w+)?)\s+(.+)', 'partial'),
        # Split after "who" (relative clause)
        (r'(.*\bwho\s+\w+)\s+(.+)', 'partial'),
        # Split after "that" (relative clause)
        (r'(.*\bthat\s+\w+)\s+(.+)', 'partial'),
        # Split after conjunctions
        (r'(.*)\s+\b(and|but|or)\s+(.+)', 'conjunction'),
        # Split at comma if it's in the middle third of the text
        (r'(.+),\s+(.+)', 'comma'),
    ]
    
    best_split = None
    best_score = -1
    
    for pattern, split_type in split_patterns:
        match = re.search(pattern, bio_text, re.IGNORECASE)
        if match:
            if split_type == 'conjunction':
                # For conjunctions, keep the conjunction with the prompt
                prompt = match.group(1) + ' ' + match.group(2)
                completion = match.group(3)
            elif split_type == 'comma':
                prompt = match.group(1)
                completion = match.group(2)
                # Only use comma splits if they're in a good position
                prompt_words = len(prompt.split())
                total_words = len(bio_text.split())
                if prompt_words < total_words * 0.3 or prompt_words > total_words * 0.7:
                    continue
            else:
                prompt = match.group(1)
                completion = match.group(2)
            
            # Score based on balance and naturalness
            prompt_words = len(prompt.split())
            completion_words = len(completion.split())
            
            # Skip if prompt is too short or too long
            if prompt_words < 5 or prompt_words > 30:
                continue
            if completion_words < 5:
                continue
                
            # Calculate score (prefer balanced splits)
            balance = min(prompt_words, completion_words) / max(prompt_words, completion_words)
            
            # Bonus for certain split types
            type_bonus = {
                'partial': 1.2,  # Prefer incomplete thoughts
                'conjunction': 1.1,
                'comma': 1.0
            }
            
            score = balance * type_bonus.get(split_type, 1.0)
            
            # Penalty if prompt ends with punctuation
            if prompt.rstrip()[-1] in '.!?':
                score *= 0.5
                
            if score > best_score:
                best_score = score
                best_split = (prompt.strip(), completion.strip())
    
    # Fallback: split at roughly 40% if no pattern matches
    if not best_split:
        words = bio_text.split()
        if len(words) >= 10:
            split_idx = int(len(words) * 0.4)
            # Find the next word boundary that isn't punctuation
            while split_idx < len(words) - 5 and words[split_idx-1][-1] in '.!?,':
                split_idx += 1
            prompt = ' '.join(words[:split_idx])
            completion = ' '.join(words[split_idx:])
            if len(prompt.split()) >= 5 and len(completion.split()) >= 5:
                best_split = (prompt, completion)
    
    return best_split

def prepare_mlx_data():
    """Convert bio.json to MLX training format"""
    
    # Load bio data
    with open('../data/bio.json', 'r') as f:
        bios = json.load(f)
    
    print(f"Loaded {len(bios)} bios")
    
    # Create prompt-completion pairs
    training_data = []
    skipped = 0
    
    for bio in bios:
        result = create_natural_split(bio)
        if result:
            prompt, completion = result
            # MLX format with chat template
            text = f"<|user|>\nComplete this bio: {prompt}<|end|>\n<|assistant|>\n{completion}<|end|>"
            training_data.append({"text": text})
        else:
            skipped += 1
    
    print(f"Created {len(training_data)} training examples")
    print(f"Skipped {skipped} bios (too short or couldn't split)")
    
    # Shuffle data
    random.seed(42)
    random.shuffle(training_data)
    
    # Split into train/val/test
    total = len(training_data)
    train_size = int(total * 0.8)
    val_size = int(total * 0.1)
    
    train_data = training_data[:train_size]
    val_data = training_data[train_size:train_size + val_size]
    test_data = training_data[train_size + val_size:]
    
    # Save to JSONL files
    output_dir = Path('bio_mlx_improved')
    output_dir.mkdir(exist_ok=True)
    
    for name, data in [('train', train_data), ('valid', val_data), ('test', test_data)]:
        with open(output_dir / f'{name}.jsonl', 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    
    # Save example prompts
    with open(output_dir / 'examples.txt', 'w') as f:
        f.write("Sample training examples:\n")
        f.write("=" * 60 + "\n\n")
        for i, item in enumerate(train_data[:5]):
            text = item['text']
            # Extract prompt and completion for display
            prompt_match = re.search(r'Complete this bio: (.+?)<\|end\|>', text)
            completion_match = re.search(r'<\|assistant\|>\n(.+?)<\|end\|>', text)
            if prompt_match and completion_match:
                f.write(f"Example {i+1}:\n")
                f.write(f"Prompt: {prompt_match.group(1)}\n")
                f.write(f"Completion: {completion_match.group(1)}\n")
                f.write("-" * 60 + "\n\n")
    
    # Print statistics
    print("\nData split:")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Valid: {len(val_data)} examples")
    print(f"  Test: {len(test_data)} examples")
    
    # Analyze prompt/completion lengths
    prompt_lengths = []
    completion_lengths = []
    
    for item in train_data[:100]:  # Sample first 100
        text = item['text']
        prompt_match = re.search(r'Complete this bio: (.+?)<\|end\|>', text)
        completion_match = re.search(r'<\|assistant\|>\n(.+?)<\|end\|>', text)
        if prompt_match and completion_match:
            prompt_lengths.append(len(prompt_match.group(1).split()))
            completion_lengths.append(len(completion_match.group(1).split()))
    
    if prompt_lengths:
        print(f"\nPrompt length: avg {sum(prompt_lengths)/len(prompt_lengths):.1f} words")
        print(f"Completion length: avg {sum(completion_lengths)/len(completion_lengths):.1f} words")
    
    print(f"\nData saved to: {output_dir}")
    print("\nNext step: Run ./start_training.sh to train the model")

if __name__ == "__main__":
    prepare_mlx_data()
EOF

# Activate virtual environment and run the script
echo "Activating Python environment..."
cd python
source venv/bin/activate

echo ""
echo "Converting bio.json to MLX format..."
echo ""

python mlx_training/prepare_bio_mlx_improved.py

# Check if conversion was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Data preparation completed successfully!"
    echo ""
    echo "Output files:"
    echo "  - $OUTPUT_DIR/train.jsonl"
    echo "  - $OUTPUT_DIR/valid.jsonl"
    echo "  - $OUTPUT_DIR/test.jsonl"
    echo "  - $OUTPUT_DIR/examples.txt"
    echo ""
    echo "Next steps:"
    echo "  1. Review examples: cat $OUTPUT_DIR/examples.txt"
    echo "  2. Start training: ./start_training.sh"
else
    echo ""
    echo "❌ Data preparation failed. Check the error messages above."
    exit 1
fi

# Deactivate virtual environment
deactivate
cd ..

echo ""
echo "🎉 Training data ready!"
echo ""