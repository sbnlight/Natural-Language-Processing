#!/usr/bin/env python3
"""
Script to evaluate multiple checkpoints and find the best one based on Validation Accuracy.
"""

import json
import torch
import sys
import glob
import re
import pickle
from pathlib import Path
from typing import List, Dict, Any

# Add parent path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from part2.model import TransformerLM
from part1.tokenizer import get_tokenizer
from part4.prompting import PromptTemplate, PromptingPipeline, evaluate_prompting

# =============================================================================
# CONFIGURATION
# =============================================================================

# [Config]: Must match the configuration used in train_baseline.py
# Based on your logs (LR ~2e-5), it seems you are using the 'medium' config.
MODEL_CONFIG = {
    "vocab_size": 8192,      # Medium: 8192, Small: 4096
    "d_model": 512,          # Medium: 512,  Small: 256
    "num_layers": 8,         # Medium: 8,    Small: 6
    "num_heads": 8,          # Medium: 8,    Small: 8
    "d_ff": 2048,            # Medium: 2048, Small: 1024
    "context_length": 512,   # Standard for this assignment
}

# Paths
BASE_DIR = Path(__file__).parent.parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
QA_DEV_PATH = BASE_DIR / "part4" / "fixtures" / "squad_dev.json"
TOKENIZER_CACHE = BASE_DIR / "part4" / "fixtures" / f"tokenizer_cache_{MODEL_CONFIG['vocab_size']}.pkl"

# =============================================================================
# UTILS
# =============================================================================

def load_tokenizer_from_cache():
    """Load the pre-trained tokenizer from the pickle cache."""
    print(f"Loading tokenizer from {TOKENIZER_CACHE}...")
    if not TOKENIZER_CACHE.exists():
        raise FileNotFoundError(f"Tokenizer cache not found at {TOKENIZER_CACHE}. Please run training first.")
    
    with open(TOKENIZER_CACHE, 'rb') as f:
        vocab, merges = pickle.load(f)
    
    special_tokens = ["<|endoftext|>", "<|pad|>"]
    return get_tokenizer(vocab, merges, special_tokens)

def get_sorted_checkpoints(directory: Path, prefix: str = "finetune") -> List[Path]:
    """Find and sort checkpoints by epoch number."""
    files = list(directory.glob(f"{prefix}_epoch_*.pt"))
    
    def extract_epoch(path):
        match = re.search(r"epoch_(\d+).pt", str(path))
        return int(match.group(1)) if match else -1
        
    return sorted(files, key=extract_epoch)

# =============================================================================
# MAIN
# =============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Data and Tokenizer
    print("Loading Validation Data...")
    with open(QA_DEV_PATH, "r") as f:
        dev_data = json.load(f)
    print(f"Found {len(dev_data)} validation examples.")

    tokenizer = load_tokenizer_from_cache()

    # 2. Initialize Model Structure (Weights will be loaded later)
    print("Initializing Model Structure...")
    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        context_length=MODEL_CONFIG["context_length"],
        d_model=MODEL_CONFIG["d_model"],
        num_layers=MODEL_CONFIG["num_layers"],
        num_heads=MODEL_CONFIG["num_heads"],
        d_ff=MODEL_CONFIG["d_ff"],
    ).to(device)

    # 3. Find Checkpoints
    checkpoints = get_sorted_checkpoints(CHECKPOINT_DIR, prefix="finetune")
    if not checkpoints:
        print(f"No checkpoints found in {CHECKPOINT_DIR}")
        return

    print(f"\nFound {len(checkpoints)} checkpoints to evaluate.")

    # 4. Evaluation Loop
    results = {}
    best_acc = 0.0
    best_epoch = -1

    # Use "simple" template. By default, PromptTemplate adds Few-Shot examples
    # if choice_format is "letter". This matches the Prompting logic.
    template = PromptTemplate(template_name="simple")

    print("\n" + "="*60)
    print("STARTING EVALUATION")
    print("="*60)

    for ckpt_path in checkpoints:
        epoch_num = re.search(r"epoch_(\d+).pt", str(ckpt_path)).group(1)
        print(f"\nEvaluating Epoch {epoch_num} (File: {ckpt_path.name})...")
        
        # Load weights
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        
        # Create Pipeline
        pipeline = PromptingPipeline(
            model=model,
            tokenizer=tokenizer,
            template=template,
            device=device
        )
        
        # Run Evaluation
        # Note: Using batch_size=8 to speed it up
        metrics = evaluate_prompting(pipeline, dev_data, batch_size=8)
        acc = metrics["accuracy"]
        
        results[epoch_num] = acc
        print(f"--> Epoch {epoch_num} Accuracy: {acc:.2%}")
        
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch_num

    # 5. Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for ep, acc in results.items():
        print(f"Epoch {ep}: {acc:.2%}")
        
    print(f"\nBEST MODEL: Epoch {best_epoch} with {best_acc:.2%} Accuracy")
    print("="*60)

if __name__ == "__main__":
    main()