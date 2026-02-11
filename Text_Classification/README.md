# Text Classification

CS288 (Natural Language Processing) Assignment 1 — UC Berkeley, Spring 2026.

## Overview

Two-part assignment covering language modeling and text classification from scratch.

### Part 1 — N-gram Language Models (`Part1_Language_Modeling/`)

Implements and evaluates bigram, trigram, and neural n-gram language models on a text corpus. Explores perplexity as an evaluation metric and compares smoothing strategies.

Run in Jupyter: `Part1.ipynb`

### Part 2 — Text Classification (`Part2_Text_Classification/`)

Implements a **Perceptron** and a **Multilayer Perceptron (MLP)** classifier trained on sparse text features. Evaluated on two datasets:

- **20 Newsgroups** — 20-class topic classification
- **SST-2** — Binary sentiment analysis (Stanford Sentiment Treebank)

**Feature types** (combinable with `+`):

| Flag | Description |
|------|-------------|
| `bow` | Bag of Words (binary) |
| `bow_counts` | Bag of Words with term frequency |
| `bigram` | Bigram features |
| `trigram` | Trigram features |
| `len` | Sentence length buckets |
| `sentiment` | Positive/negative word counts |
| `punct` | Punctuation features |
| `caps` | Capitalisation ratio |
| `wordstats` | Average word length, long/short word counts |

## Setup

```bash
conda create -n cs288a1 python=3.10
conda activate cs288a1
pip install -r Part2_Text_Classification/requirements.txt
```

## Usage

```bash
cd Part2_Text_Classification

# Train Perceptron
python perceptron.py -d newsgroups -f bow
python perceptron.py -d sst2 -f bow+sentiment+len

# Train MLP
python multilayer_perceptron.py -d newsgroups
python multilayer_perceptron.py -d sst2

# Run tests
pytest
```

## File Structure

```
Part1_Language_Modeling/
└── Part1.ipynb                  # N-gram language model notebook

Part2_Text_Classification/
├── features.py                  # Feature extraction classes
├── perceptron.py                # Perceptron classifier
├── multilayer_perceptron.py     # MLP classifier
├── utils.py                     # Data loading and evaluation utilities
├── stopwords.txt                # Stopword list for BoW features
├── data/
│   ├── newsgroups/              # 20 Newsgroups dataset (train/dev/test)
│   └── sst2/                   # SST-2 sentiment dataset (train/dev/test)
├── results/                     # Model predictions
└── tests/                       # Unit tests
```
