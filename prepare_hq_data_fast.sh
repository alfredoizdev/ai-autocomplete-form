#!/bin/bash

# Fast high-quality data preparation

# Check for command line argument
if [ $# -eq 0 ]; then
    echo "Usage: ./prepare_hq_data_fast.sh [csv_file] [output_dir]"
    echo "Example: ./prepare_hq_data_fast.sh data/bios100000.csv bios100k_hq"
    echo "Using default: data/newBios20000.csv -> lookingfor_hq/"
    CSV_FILE="../../data/newBios20000.csv"
    OUTPUT_DIR="lookingfor_hq"
else
    # Convert relative path to absolute path if needed
    if [[ "$1" == /* ]]; then
        CSV_FILE="$1"
    else
        CSV_FILE="../../$1"
    fi
    OUTPUT_DIR="${2:-lookingfor_hq}"
fi

echo "Preparing high-quality training data (fast version)..."
echo "Input file: $CSV_FILE"
echo "Output directory: $OUTPUT_DIR"

# Change to MLX training directory
cd python/mlx_training

# Activate virtual environment
source ../venv/bin/activate

# Run the fast data preparation with arguments
echo "Processing dataset with quality filters..."
python prepare_lookingfor_mlx_fast.py "$CSV_FILE" "$OUTPUT_DIR"

echo "Data preparation complete!"
echo "High-quality dataset saved to: python/mlx_training/$OUTPUT_DIR/"

# Show statistics
echo -e "\nDataset statistics:"
echo "Train samples: $(wc -l < $OUTPUT_DIR/train.jsonl)"
echo "Valid samples: $(wc -l < $OUTPUT_DIR/valid.jsonl)"
echo "Test samples: $(wc -l < $OUTPUT_DIR/test.jsonl)"