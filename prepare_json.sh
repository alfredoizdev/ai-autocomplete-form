#!/bin/bash

# prepare_json.sh - Convert CSV files to JSON format in the data folder

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if a CSV file is provided as argument
if [ $# -eq 0 ]; then
    print_error "No CSV file specified"
    echo "Usage: $0 <csv_filename>"
    echo "Example: $0 newBios20000.csv"
    echo ""
    echo "Available CSV files in data folder:"
    ls data/*.csv 2>/dev/null || echo "  No CSV files found in data folder"
    exit 1
fi

CSV_FILE="$1"

# Check if the CSV file exists in data folder
if [ ! -f "data/$CSV_FILE" ]; then
    print_error "CSV file 'data/$CSV_FILE' not found"
    echo ""
    echo "Available CSV files in data folder:"
    ls data/*.csv 2>/dev/null || echo "  No CSV files found in data folder"
    exit 1
fi

# Get the base filename without extension
BASE_NAME="${CSV_FILE%.*}"
JSON_FILE="${BASE_NAME}.json"

print_info "Converting CSV to JSON..."
print_info "Input: data/$CSV_FILE"
print_info "Output: data/$JSON_FILE"

# Create Python script for conversion
python3 - "$CSV_FILE" "$JSON_FILE" << 'EOF'
import csv
import json
import sys
from pathlib import Path

def convert_csv_to_json(csv_filename, json_filename):
    """Convert CSV file to JSON format."""
    csv_path = Path("data") / csv_filename
    json_path = Path("data") / json_filename
    
    try:
        # Load CSV data
        bios = []
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0].strip():  # Skip empty rows
                    bio = row[0].strip()
                    # Remove surrounding quotes if present
                    if bio.startswith('"') and bio.endswith('"'):
                        bio = bio[1:-1]
                    # Replace double quotes with single quotes
                    bio = bio.replace('""', '"')
                    bios.append(bio)
        
        # Save as JSON with pretty formatting
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(bios, f, indent=2, ensure_ascii=False)
        
        # Print statistics
        print(f"\nSuccessfully converted {len(bios)} entries")
        print(f"Output saved to: {json_path}")
        
        # Calculate stats
        if bios:
            word_counts = [len(bio.split()) for bio in bios]
            avg_words = sum(word_counts) / len(word_counts)
            print(f"\nStatistics:")
            print(f"  Total entries: {len(bios)}")
            print(f"  Average length: {avg_words:.1f} words")
            print(f"  Min/Max words: {min(word_counts)}/{max(word_counts)}")
            
            # Show first few examples
            print(f"\nFirst 3 examples:")
            for i, bio in enumerate(bios[:3], 1):
                preview = bio[:80] + "..." if len(bio) > 80 else bio
                print(f"  {i}. {preview}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

# Get arguments from bash
csv_file = sys.argv[1]
json_file = sys.argv[2]

# Run conversion
exit_code = convert_csv_to_json(csv_file, json_file)
sys.exit(exit_code)
EOF

# Check if conversion was successful
if [ $? -eq 0 ]; then
    print_info "Conversion completed successfully!"
    
    # Check if the JSON file was created
    if [ -f "data/$JSON_FILE" ]; then
        FILE_SIZE=$(ls -lh "data/$JSON_FILE" | awk '{print $5}')
        print_info "JSON file created: data/$JSON_FILE (size: $FILE_SIZE)"
    fi
else
    print_error "Conversion failed!"
    exit 1
fi