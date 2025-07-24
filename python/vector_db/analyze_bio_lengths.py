import json
from pathlib import Path
import statistics

# Load bio.json
bio_path = Path(__file__).parent.parent.parent / "data" / "bio.json"
with open(bio_path, 'r') as f:
    bios = json.load(f)

# Analyze lengths
lengths = [len(bio.split()) for bio in bios]
char_lengths = [len(bio) for bio in bios]

print(f"Total bios: {len(bios)}")
print(f"\nWord count statistics:")
print(f"  Min: {min(lengths)} words")
print(f"  Max: {max(lengths)} words")
print(f"  Mean: {statistics.mean(lengths):.1f} words")
print(f"  Median: {statistics.median(lengths)} words")

# Show distribution
print(f"\nWord count distribution:")
print(f"  < 20 words: {sum(1 for l in lengths if l < 20)}")
print(f"  20-50 words: {sum(1 for l in lengths if 20 <= l < 50)}")
print(f"  50-100 words: {sum(1 for l in lengths if 50 <= l < 100)}")
print(f"  100-200 words: {sum(1 for l in lengths if 100 <= l < 200)}")
print(f"  200+ words: {sum(1 for l in lengths if l >= 200)}")

# Show some very long examples
print(f"\nLongest bios (by word count):")
sorted_bios = sorted(zip(bios, lengths), key=lambda x: x[1], reverse=True)
for bio, length in sorted_bios[:5]:
    print(f"\n{length} words: {bio[:100]}...")

# Check for problematic patterns
print(f"\n\nProblematic patterns found:")
multi_newline_count = sum(1 for bio in bios if '\n\n' in bio)
print(f"  Bios with multiple newlines: {multi_newline_count}")

long_single_sentence = sum(1 for bio in bios if len(bio.split('.')) == 1 and len(bio.split()) > 100)
print(f"  Very long single sentences (100+ words): {long_single_sentence}")