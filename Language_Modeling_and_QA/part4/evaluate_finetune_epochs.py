import torch
import sys
import os
import glob
import re
import json
import pickle
from pathlib import Path

# 1. Setup Paths (Based on your Colab path)
# 确保这个路径是正确的，指向你的作业根目录
PROJECT_ROOT = Path("/content/drive/MyDrive/CS288/Assignment_2/cs288-sp26-a2-main")
sys.path.append(str(PROJECT_ROOT))

# Imports from your project
from part4.qa_model import TransformerForMultipleChoice, evaluate_qa_model
from part2.model import TransformerLM
from part4.datasets import create_qa_dataloader
from part1.tokenizer import get_tokenizer

def run_validation_sweep():
    print("="*60)
    print("SEARCHING FOR THE BEST FINE-TUNED CHECKPOINT")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 2. Configuration (MUST match your Medium config)
    # 根据你之前的Log，Vocab是8192
    config = {
        "vocab_size": 8192,
        "d_model": 512,
        "num_layers": 8,
        "num_heads": 8,
        "d_ff": 2048,
        "context_length": 512,
        "batch_size": 8, # Inference uses less memory, so 16 is fine
    }

    # 3. Load Tokenizer (Use cache for speed)
    cache_path = PROJECT_ROOT / "part4" / "fixtures" / f"tokenizer_cache_{config['vocab_size']}.pkl"
    print(f"\nLoading tokenizer from {cache_path}...")
    with open(cache_path, 'rb') as f:
        vocab, merges = pickle.load(f)
    tokenizer = get_tokenizer(vocab, merges, ["<|endoftext|>", "<|pad|>"])

    # 4. Load Validation Data
    val_path = PROJECT_ROOT / "part4" / "fixtures" / "squad_dev.json"
    print(f"Loading validation data from {val_path}...")
    with open(val_path) as f:
        dev_data = json.load(f)
    
    # Create DataLoader (Shuffle=False is CRITICAL for consistent evaluation)
    dev_dataloader = create_qa_dataloader(
        data=dev_data,
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_length=config["context_length"],
        num_choices=4,
        shuffle=False, 
    )

    # 5. Initialize Model Structure (Once)
    print("Initializing model architecture...")
    pretrained_model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    )
    model = TransformerForMultipleChoice(
        transformer_lm=pretrained_model,
        hidden_size=config["d_model"],
        num_choices=4,
        pooling="last"
    ).to(device)

    # 6. Find all checkpoints
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    # Find files matching finetune_gen_epoch_*.pt
    ckpt_files = list(ckpt_dir.glob("finetune_gen_epoch_*.pt"))
    
    if not ckpt_files:
        print(f"No checkpoints found in {ckpt_dir}!")
        return

    # Sort files by epoch number (integer), not string
    # extracts '5' from 'finetune_gen_epoch_5.pt'
    def extract_epoch(path):
        match = re.search(r"epoch_(\d+)", str(path))
        return int(match.group(1)) if match else -1
    
    ckpt_files.sort(key=extract_epoch)

    print(f"\nFound {len(ckpt_files)} checkpoints. Starting evaluation...\n")
    print(f"{'Epoch':<10} | {'Accuracy':<10}")
    print("-" * 25)

    # 7. Evaluate Loop
    best_acc = 0.0
    best_epoch = -1

    # Keep evaluating without calculating gradients
    with torch.no_grad():
        for ckpt_path in ckpt_files:
            epoch_num = extract_epoch(ckpt_path)
            
            # Load weights
            # map_location ensures we don't run OOM if saving on different device
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            
            # Evaluate
            results = evaluate_qa_model(model, dev_dataloader, device)
            acc = results['accuracy']
            
            print(f"{epoch_num:<10} | {acc:.2%}")
            
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch_num

    print("-" * 25)
    print(f"\n🏆 BEST CHECKPOINT: Epoch {best_epoch} with Accuracy {best_acc:.2%}")
    print(f"File: {ckpt_dir}/finetune_gen_epoch_{best_epoch}.pt")

# Run it
if __name__ == "__main__":
    # Clean memory first just in case
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    run_validation_sweep()