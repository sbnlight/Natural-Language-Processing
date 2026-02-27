#!/bin/bash
# =============================================================================
# CS288 Assignment 2 - Submission Creator
# =============================================================================
# This script creates a submission.zip file containing only the files needed
# for grading. Run this from the source/ directory.
#
# Usage:
#   ./create_submission.sh
#
# Output:
#   submission.zip in the current directory
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "CS288 Assignment 2 - Submission Creator"
echo "========================================"
echo

# Check we're in the right directory
if [[ ! -d "part1" ]] || [[ ! -d "part2" ]] || [[ ! -d "part3" ]]; then
    echo -e "${RED}Error: Please run this script from the source/ directory${NC}"
    echo "Expected to find part1/, part2/, part3/, part4/ directories"
    exit 1
fi

# Remove old submission if exists
if [[ -f "submission.zip" ]]; then
    echo -e "${YELLOW}Removing old submission.zip...${NC}"
    rm submission.zip
fi

# Create temporary directory for submission
TEMP_DIR=$(mktemp -d)
echo "Creating submission structure..."

# Part 1: Tokenization
echo "  - Adding part1/ (BPE tokenizer)..."
mkdir -p "$TEMP_DIR/part1"
cp part1/train_bpe.py "$TEMP_DIR/part1/"
cp part1/tokenizer.py "$TEMP_DIR/part1/"
touch "$TEMP_DIR/part1/__init__.py"

# Part 2: Transformer Model
echo "  - Adding part2/ (Transformer model)..."
mkdir -p "$TEMP_DIR/part2"
cp part2/model.py "$TEMP_DIR/part2/"
touch "$TEMP_DIR/part2/__init__.py"

# Part 3: Neural Network Utilities
echo "  - Adding part3/ (NN utilities)..."
mkdir -p "$TEMP_DIR/part3"
cp part3/nn_utils.py "$TEMP_DIR/part3/"
touch "$TEMP_DIR/part3/__init__.py"

# Part 4: Training and Evaluation (if exists)
if [[ -d "part4" ]]; then
    echo "  - Adding part4/ (Training & QA)..."
    mkdir -p "$TEMP_DIR/part4"
    [[ -f "part4/datasets.py" ]] && cp part4/datasets.py "$TEMP_DIR/part4/"
    [[ -f "part4/sampling.py" ]] && cp part4/sampling.py "$TEMP_DIR/part4/"
    [[ -f "part4/prompting.py" ]] && cp part4/prompting.py "$TEMP_DIR/part4/"
    [[ -f "part4/qa_model.py" ]] && cp part4/qa_model.py "$TEMP_DIR/part4/"
    [[ -f "part4/trainer.py" ]] && cp part4/trainer.py "$TEMP_DIR/part4/"
    touch "$TEMP_DIR/part4/__init__.py"
    
    # Part 4 prediction outputs (required for grading)
    if [[ -d "part4/outputs" ]]; then
        echo "  - Adding part4/outputs/ (prediction files)..."
        mkdir -p "$TEMP_DIR/part4/outputs"
        [[ -f "part4/outputs/finetuned_predictions.json" ]] && cp part4/outputs/finetuned_predictions.json "$TEMP_DIR/part4/outputs/"
        [[ -f "part4/outputs/prompting_predictions.json" ]] && cp part4/outputs/prompting_predictions.json "$TEMP_DIR/part4/outputs/"
    else
        echo -e "${YELLOW}  Warning: part4/outputs/ not found. Run train_baseline.py first to generate predictions.${NC}"
    fi
fi

# Create the zip file
echo
echo "Creating submission.zip..."
cd "$TEMP_DIR"
zip -r submission.zip part1/ part2/ part3/ part4/ 2>/dev/null || zip -r submission.zip part1/ part2/ part3/
mv submission.zip "$OLDPWD/"
cd "$OLDPWD"

# Cleanup
rm -rf "$TEMP_DIR"

# Verify the submission
echo
echo "========================================"
echo -e "${GREEN}SUCCESS: submission.zip created!${NC}"
echo "========================================"
echo
echo "Contents:"
unzip -l submission.zip | grep -E "\.(py|json)$" | awk '{print "  " $4}'
echo
echo "File size: $(du -h submission.zip | cut -f1)"
echo
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Upload submission.zip to Gradescope"
echo "  2. Wait for autograder results"
echo
