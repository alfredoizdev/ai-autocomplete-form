"""
Fix capitalization issues in training data.
Specifically fixes lowercase 'i' and ensures proper sentence capitalization.
"""

import json
import re
from typing import Dict
import sys

def fix_capitalization(text: str) -> str:
    """Fix common capitalization issues in text."""
    # Fix standalone lowercase 'i' to 'I'
    text = re.sub(r'\bi\b', 'I', text)
    
    # Fix "i'm", "i'll", "i've", "i'd" etc.
    text = re.sub(r'\bi\'', 'I\'', text)
    
    # Ensure first letter of completion is capitalized if it starts a sentence
    # (unless it's a continuation word like 'and', 'but', 'or')
    words = text.split()
    if words:
        first_word = words[0].lower()
        continuation_words = {'and', 'but', 'or', 'who', 'which', 'that', 'where', 'when', 
                             'with', 'to', 'for', 'in', 'about', 'from'}
        
        # If not a continuation word and starts with lowercase letter, capitalize it
        if first_word not in continuation_words and words[0][0].islower():
            words[0] = words[0][0].upper() + words[0][1:]
            text = ' '.join(words)
    
    # Fix sentence starts after punctuation
    text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    
    return text

def process_jsonl_file(input_file: str, output_file: str):
    """Process a JSONL file and fix capitalization issues."""
    fixed_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                total_count += 1
                try:
                    data = json.loads(line.strip())
                    
                    # Extract the completion part from the text field
                    if 'text' in data:
                        text = data['text']
                        
                        # Find the assistant response part
                        assistant_match = re.search(r'<\|assistant\|>\n(.*?)<\|end\|>', text, re.DOTALL)
                        if assistant_match:
                            original_completion = assistant_match.group(1)
                            fixed_completion = fix_capitalization(original_completion)
                            
                            if original_completion != fixed_completion:
                                fixed_count += 1
                                # Replace the completion in the text
                                text = text.replace(
                                    f'<|assistant|>\n{original_completion}<|end|>',
                                    f'<|assistant|>\n{fixed_completion}<|end|>'
                                )
                                data['text'] = text
                    
                    # Write the (possibly modified) data
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    
                except Exception as e:
                    print(f"Error processing line {total_count}: {e}")
                    # Write original line if there's an error
                    outfile.write(line)
    
    print(f"Processed {total_count} lines, fixed {fixed_count} capitalization issues")
    return fixed_count, total_count

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fix_capitalization.py <input_file> <output_file>")
        print("Example: python fix_capitalization.py train.jsonl train_fixed.jsonl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Fixing capitalization in {input_file}...")
    fixed, total = process_jsonl_file(input_file, output_file)
    print(f"Done! Fixed {fixed}/{total} lines ({fixed/total*100:.1f}%)")
    print(f"Output saved to: {output_file}")