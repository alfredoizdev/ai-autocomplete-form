#!/usr/bin/env python3
"""
Extract quality sentences from bio.json for better training data.
Filters for grammatically correct sentences with 8-20 words.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

def split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex patterns."""
    # Handle common abbreviations
    text = text.replace('Mr.', 'Mr')
    text = text.replace('Mrs.', 'Mrs')
    text = text.replace('Dr.', 'Dr')
    text = text.replace('Ms.', 'Ms')
    text = text.replace('St.', 'St')
    text = text.replace('etc.', 'etc')
    text = text.replace('i.e.', 'ie')
    text = text.replace('e.g.', 'eg')
    
    # Split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Additional split on common patterns
    expanded = []
    for sent in sentences:
        # Split very long sentences on ", and" or ", but"
        if len(sent) > 100:
            parts = re.split(r',\s+(?:and|but)\s+', sent)
            if len(parts) > 1:
                for part in parts:
                    if part.strip():
                        expanded.append(part.strip())
            else:
                expanded.append(sent)
        else:
            expanded.append(sent)
    
    return [s.strip() for s in expanded if s.strip()]

def is_grammatically_correct(sentence: str) -> bool:
    """
    Check if a sentence appears to be grammatically correct.
    Basic checks for capitalization, punctuation, and structure.
    """
    sentence = sentence.strip()
    
    # Must have content
    if not sentence or len(sentence) < 15:
        return False
    
    # Should start with capital letter or quote
    if not sentence[0].isupper() and not sentence.startswith('"') and not sentence.startswith("'"):
        return False
    
    # Should end with proper punctuation
    if not re.match(r'.*[.!?"\']$', sentence):
        return False
    
    # Should not have multiple spaces
    if '  ' in sentence:
        return False
    
    # Should not be all caps (except short words)
    words = sentence.split()
    caps_words = [w for w in words if w.isupper() and len(w) > 3]
    if len(caps_words) > len(words) * 0.3:  # More than 30% all caps
        return False
    
    # Must contain at least one verb or verb-like pattern
    verb_patterns = [
        r'\b(am|is|are|was|were|been|being|be)\b',
        r'\b(have|has|had|having)\b',
        r'\b(do|does|did|doing|done)\b',
        r'\b(will|would|shall|should|can|could|may|might|must)\b',
        r'\b(want|wants|wanted|wanting)\b',
        r'\b(need|needs|needed|needing)\b',
        r'\b(like|likes|liked|liking)\b',
        r'\b(love|loves|loved|loving)\b',
        r'\b(enjoy|enjoys|enjoyed|enjoying)\b',
        r'\b(seek|seeks|seeking|sought)\b',
        r'\b(look|looks|looking|looked)\b',
        r'\b\w+ing\b',  # gerunds
        r'\b\w+ed\b'    # past tense
    ]
    
    has_verb = any(re.search(pattern, sentence.lower()) for pattern in verb_patterns)
    if not has_verb:
        return False
    
    # Check for basic sentence structure (has noun-like and verb-like words)
    has_noun = any(word[0].isupper() or word.lower() in ['i', 'we', 'they', 'he', 'she', 'it', 'you'] 
                   for word in words)
    
    return has_noun

def clean_sentence(sentence: str) -> str:
    """Clean up a sentence while preserving meaning."""
    # Remove extra whitespace
    sentence = ' '.join(sentence.split())
    
    # Fix spacing around punctuation
    sentence = re.sub(r'\s+([,\.!?;:])', r'\1', sentence)
    sentence = re.sub(r'([,\.!?;:])\s*([,\.!?;:])', r'\1\2', sentence)
    
    # Fix quotes
    if sentence.count('"') == 1:
        sentence = sentence.replace('"', '')
    if sentence.count("'") == 1 and not re.search(r"\b\w+'\w+\b", sentence):
        sentence = sentence.replace("'", '')
    
    # Ensure proper ending punctuation
    if sentence and not sentence[-1] in '.!?"\'':
        # Add period if it seems like a complete thought
        if len(sentence.split()) >= 8:
            sentence += '.'
    
    return sentence.strip()

def extract_quality_sentences(text: str) -> List[str]:
    """Extract quality sentences from bio text."""
    if not text:
        return []
    
    # Clean the text
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    text = ' '.join(text.split())
    
    # Split into sentences
    sentences = split_sentences(text)
    
    quality_sentences = []
    for sent in sentences:
        # Clean the sentence
        cleaned = clean_sentence(sent)
        
        # Check if grammatically correct
        if not is_grammatically_correct(cleaned):
            continue
        
        # Check word count
        words = cleaned.split()
        if 8 <= len(words) <= 20:
            quality_sentences.append(cleaned)
    
    return quality_sentences

def process_bio_data(input_file: Path, output_file: Path) -> Tuple[int, int, List[Dict]]:
    """Process bio data and extract quality sentences."""
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} bios...")
    
    all_sentences = []
    total_sentences_found = 0
    
    for i, bio_text in enumerate(data):
        if i % 500 == 0:
            print(f"  Processed {i}/{len(data)} bios...")
        
        if not bio_text or not isinstance(bio_text, str):
            continue
        
        # Extract quality sentences
        sentences = extract_quality_sentences(bio_text)
        total_sentences_found += len(sentences)
        
        for sentence in sentences:
            all_sentences.append({
                'sentence': sentence,
                'word_count': len(sentence.split()),
                'original_bio_id': f'bio_{i}'
            })
    
    # Remove duplicates
    print("Removing duplicates...")
    seen = set()
    unique_sentences = []
    for item in all_sentences:
        sentence_lower = item['sentence'].lower()
        if sentence_lower not in seen:
            seen.add(sentence_lower)
            unique_sentences.append(item)
    
    # Sort by quality (prefer medium length sentences)
    unique_sentences.sort(key=lambda x: abs(x['word_count'] - 14))
    
    # Save the processed data
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_sentences, f, indent=2, ensure_ascii=False)
    
    return total_sentences_found, len(unique_sentences), unique_sentences

def analyze_results(sentences: List[Dict]) -> None:
    """Analyze and print statistics about the extracted sentences."""
    print("\n" + "="*60)
    print("ANALYSIS OF EXTRACTED SENTENCES")
    print("="*60)
    
    print(f"\nTotal quality sentences: {len(sentences)}")
    
    # Word count distribution
    word_counts = [s['word_count'] for s in sentences]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    
    print(f"Average words per sentence: {avg_words:.1f}")
    print(f"Min words: {min(word_counts) if word_counts else 0}")
    print(f"Max words: {max(word_counts) if word_counts else 0}")
    
    # Word count histogram
    print("\nWord Count Distribution:")
    count_dist = Counter(word_counts)
    for wc in range(8, 21):
        count = count_dist.get(wc, 0)
        bar = '█' * (count // 50)
        print(f"{wc:2d} words: {count:4d} {bar}")
    
    # Sample sentences by word count
    print("\n" + "="*60)
    print("SAMPLE SENTENCES BY WORD COUNT")
    print("="*60)
    
    for word_count in [8, 10, 12, 14, 16, 18, 20]:
        matching = [s for s in sentences if s['word_count'] == word_count]
        if matching:
            import random
            sample = random.choice(matching)
            print(f"\n{word_count} words: {sample['sentence']}")
    
    # Common patterns
    print("\n" + "="*60)
    print("COMMON STARTING PATTERNS")
    print("="*60)
    
    starts = Counter()
    for s in sentences:
        first_three = ' '.join(s['sentence'].split()[:3])
        starts[first_three] += 1
    
    print("\nMost common beginnings:")
    for pattern, count in starts.most_common(15):
        print(f"  '{pattern}...' - {count} times")
    
    # Quality examples
    print("\n" + "="*60)
    print("HIGH QUALITY EXAMPLES (varied sentence structures)")
    print("="*60)
    
    # Find diverse sentences
    diverse_samples = []
    seen_starts = set()
    
    for s in sentences:
        first_word = s['sentence'].split()[0].lower()
        if first_word not in seen_starts and 10 <= s['word_count'] <= 16:
            seen_starts.add(first_word)
            diverse_samples.append(s)
            if len(diverse_samples) >= 20:
                break
    
    for i, sample in enumerate(diverse_samples[:10], 1):
        print(f"\n{i}. {sample['sentence']}")
        print(f"   ({sample['word_count']} words)")

def main():
    """Main processing function."""
    # Set up paths
    base_dir = Path(__file__).parent.parent.parent
    input_file = base_dir / 'data' / 'bio.json'
    output_file = base_dir / 'data' / 'bio_sentences.json'
    
    print("="*60)
    print("EXTRACTING QUALITY SENTENCES FROM BIO DATA")
    print("="*60)
    print(f"\nInput: {input_file}")
    print(f"Output: {output_file}")
    print("\nCriteria:")
    print("- Grammatically correct sentences")
    print("- 8-20 words per sentence")
    print("- Proper capitalization and punctuation")
    print("- Contains verb structures")
    print("="*60)
    
    try:
        total, extracted, sentences = process_bio_data(input_file, output_file)
        
        print(f"\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Total sentences found: {total}")
        print(f"Quality sentences extracted: {extracted}")
        print(f"Extraction rate: {extracted/total*100:.1f}%" if total > 0 else "N/A")
        
        # Analyze results
        if sentences:
            analyze_results(sentences)
        
        # Create a readable sample file
        sample_file = base_dir / 'data' / 'bio_sentences_sample.txt'
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write("SAMPLE HIGH-QUALITY TRAINING SENTENCES\n")
            f.write("="*60 + "\n\n")
            f.write("These sentences are grammatically correct, 8-20 words each,\n")
            f.write("and suitable for training a language model.\n\n")
            f.write("="*60 + "\n\n")
            
            for i, s in enumerate(sentences[:100], 1):
                f.write(f"{i:3d}. {s['sentence']}\n")
        
        print(f"\n✅ Sample sentences written to: {sample_file}")
        print(f"✅ Full dataset saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()