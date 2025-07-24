import json
from pathlib import Path

# Load bio.json
bio_path = Path(__file__).parent.parent.parent / "data" / "bio.json"
with open(bio_path, 'r') as f:
    bios = json.load(f)

print(f"Original bio count: {len(bios)}")

# Filter bios:
# 1. Remove very short (< 10 words) 
# 2. Remove very long (> 100 words) - these are outliers that affect autocomplete
# 3. Keep bios between 10-100 words which are ideal for autocomplete context
filtered_bios = []
removed_counts = {
    'too_short': 0,
    'too_long': 0,
    'total_removed': 0
}

for bio in bios:
    word_count = len(bio.split())
    
    if word_count < 10:
        removed_counts['too_short'] += 1
        removed_counts['total_removed'] += 1
    elif word_count > 100:
        removed_counts['too_long'] += 1
        removed_counts['total_removed'] += 1
        print(f"Removing bio with {word_count} words: {bio[:80]}...")
    else:
        # Clean the bio text
        bio = bio.strip()
        # Remove any multiple spaces
        bio = " ".join(bio.split())
        filtered_bios.append(bio)

print(f"\nFiltering results:")
print(f"  Removed too short (< 10 words): {removed_counts['too_short']}")
print(f"  Removed too long (> 100 words): {removed_counts['too_long']}")
print(f"  Total removed: {removed_counts['total_removed']}")
print(f"  Final bio count: {len(filtered_bios)}")

# Save the cleaned data
cleaned_path = bio_path.parent / "bio_cleaned.json"
with open(cleaned_path, 'w') as f:
    json.dump(filtered_bios, f, indent=2)

print(f"\nCleaned data saved to: {cleaned_path}")

# Also update the original bio.json with the cleaned data
print("\nBacking up original bio.json...")
backup_path = bio_path.parent / "bio_original_backup.json"
with open(bio_path, 'r') as f:
    original_data = json.load(f)
with open(backup_path, 'w') as f:
    json.dump(original_data, f, indent=2)

print("Updating bio.json with cleaned data...")
with open(bio_path, 'w') as f:
    json.dump(filtered_bios, f, indent=2)

print(f"\nDone! Original data backed up to: {backup_path}")