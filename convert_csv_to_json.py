#!/usr/bin/env python3
"""
Convert LookingFor_20000.csv to JSON format for review
"""

import csv
import json
from pathlib import Path

def load_csv_data(file_path: str) -> list:
    """Load bio data from CSV file."""
    bios = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if row and row[0].strip():  # Skip empty rows
                # Remove leading/trailing quotes if present
                bio = row[0].strip()
                if bio.startswith('"') and bio.endswith('"'):
                    bio = bio[1:-1]
                # Replace double quotes with single quotes
                bio = bio.replace('""', '"')
                bios.append(bio)
    return bios

def main():
    # Paths
    csv_path = Path("data/LookingFor_20000.csv")
    json_path = Path("data/lookingfor_20000.json")
    
    print(f"Loading CSV from: {csv_path}")
    
    try:
        # Load CSV data
        bios = load_csv_data(csv_path)
        print(f"Loaded {len(bios)} bios from CSV")
        
        # Save as JSON with pretty formatting
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(bios, f, indent=2, ensure_ascii=False)
        
        print(f"\nSuccessfully converted to JSON: {json_path}")
        
        # Show some examples
        print("\nFirst 5 examples:")
        for i, bio in enumerate(bios[:5]):
            print(f"\n{i+1}. {bio[:100]}..." if len(bio) > 100 else f"\n{i+1}. {bio}")
        
        # Show statistics
        bio_lengths = [len(bio.split()) for bio in bios]
        print(f"\nStatistics:")
        print(f"  Total bios: {len(bios)}")
        print(f"  Average bio length: {sum(bio_lengths)/len(bio_lengths):.1f} words")
        print(f"  Min/Max length: {min(bio_lengths)}/{max(bio_lengths)} words")
        
        # Count multi-sentence bios
        multi_sentence = 0
        for bio in bios:
            # Simple sentence count (may not be perfect)
            sentence_count = len([s for s in bio.split('.') if s.strip()])
            if sentence_count > 1:
                multi_sentence += 1
        
        print(f"  Multi-sentence bios: {multi_sentence} ({multi_sentence/len(bios)*100:.1f}%)")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())