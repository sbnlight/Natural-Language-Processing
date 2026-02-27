# Language Modeling and QA

CS288 (Natural Language Processing) Assignment 2 — UC Berkeley, Spring 2026.

## Overview

Four-part assignment building up from tokenization to a fine-tuned question answering system.

### Part 1 — BPE Tokenizer (`part1/`)

Implements a **Byte Pair Encoding (BPE)** tokenizer compatible with GPT-2 / tiktoken. Supports encoding, decoding, and special tokens using the same byte-to-unicode mapping as GPT-2.

### Part 2 — Transformer Language Model (`part2/`)

Implements a **decoder-only Transformer** from scratch in PyTorch, including custom Linear layers (no bias, LLaMA-style), multi-head causal self-attention, RMS normalization, and feed-forward blocks.

### Part 3 — Training Utilities (`part3/`)

Implements `cross_entropy` loss and `gradient_clipping` used during model training.

### Part 4 — Question Answering (`part4/`)

Trains the language model on **SQuAD** and a custom QA dataset. Compares fine-tuning (supervised training) against prompting (few-shot in-context learning).

## Setup

```bash
conda create -n cs288a2 python=3.10
conda activate cs288a2
pip install -r requirements.txt
```

### Download Datasets

SQuAD and custom QA fixtures (~2 GB) are not included in this repo. Download with:

```bash
cd part4
python setup_datasets.py
```

## Usage

```bash
# Train baseline QA model (fine-tuning)
python part4/train_baseline.py

# Train generative model
python part4/train_baseline_gen.py

# Prompting baseline (no fine-tuning required)
python part4/prompting.py

# Evaluate checkpoints
python part4/evaluate_checkpoints.py

# Run all tests
pytest
```

## File Structure

```
part1/
├── tokenizer.py       # BPE tokenizer (GPT-2 compatible)
├── train_bpe.py       # BPE merge-pair training script
├── common.py          # GPT-2 byte-to-unicode mapping
└── tests/

part2/
├── model.py           # Decoder-only Transformer language model
└── tests/

part3/
├── nn_utils.py        # Cross-entropy loss, gradient clipping
└── tests/

part4/
├── trainer.py         # Training loop (AdamW + cosine LR schedule)
├── datasets.py        # SQuAD and QA dataset loaders
├── datasets_gen.py    # Generative dataset loaders
├── qa_model.py        # QA model architecture
├── train_baseline.py  # Fine-tuning training script
├── prompting.py       # Few-shot prompting baseline
├── evaluate_models.py # Evaluation utilities
└── outputs/           # Model predictions (JSON)
```

> **Note:** `part4/fixtures/` (SQuAD data, ~2 GB) and `checkpoints/` are excluded. Run `python part4/setup_datasets.py` to download the data.
