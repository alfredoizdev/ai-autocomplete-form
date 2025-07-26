#!/bin/bash

# Fix capitalization issues in training data

echo "Fixing capitalization issues in bios100k_hq training data..."

cd python/mlx_training

# Activate virtual environment
source ../venv/bin/activate

# Create backup directory
mkdir -p bios100k_hq_backup
cp bios100k_hq/*.jsonl bios100k_hq_backup/

# Fix each file
echo "Fixing train.jsonl..."
python fix_capitalization.py bios100k_hq/train.jsonl bios100k_hq/train_fixed.jsonl
mv bios100k_hq/train_fixed.jsonl bios100k_hq/train.jsonl

echo "Fixing valid.jsonl..."
python fix_capitalization.py bios100k_hq/valid.jsonl bios100k_hq/valid_fixed.jsonl
mv bios100k_hq/valid_fixed.jsonl bios100k_hq/valid.jsonl

echo "Fixing test.jsonl..."
python fix_capitalization.py bios100k_hq/test.jsonl bios100k_hq/test_fixed.jsonl
mv bios100k_hq/test_fixed.jsonl bios100k_hq/test.jsonl

echo "Done! Original files backed up to bios100k_hq_backup/"
echo "Now you can retrain the model with: ./start_training_mlx_community.sh"