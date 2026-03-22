# Hybrid RAG System

CS288 (Natural Language Processing) Assignment 3 — UC Berkeley, Spring 2026.

## Overview

A **Retrieval-Augmented Generation (RAG)** system for open-domain QA over the Berkeley EECS website. Built under strict autograder constraints: 2 CPUs, 3 GB RAM, no GPU, 100 questions within 60 seconds.

## Pipeline

```
Question
   │
   ├─► BM25 (sparse)  ─┐
   │                    ├─► RRF Fusion ─► Cross-Encoder Rerank ─► LLM ─► Answer
   └─► FAISS (dense)  ─┘
```

1. **Recall** — BM25 + FAISS each return top-30 candidates independently
2. **Fusion** — Reciprocal Rank Fusion (RRF, k=60) merges the two ranked lists
3. **Reranking** — Cross-encoder (`ms-marco-MiniLM-L-6-v2`) rescores top-10
4. **Generation** — Few-shot prompt → OpenRouter LLM → normalised answer

## Setup

```bash
pip install -r requirements.txt
```

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="your-key-here"
```

## Usage

```bash
# Run full pipeline
bash run.sh QA_pairs/questions.txt QA_pairs/predictions.txt

# Evaluate predictions (F1 / Exact Match)
python eval_local.py \
    --questions QA_pairs/questions.txt \
    --answers   QA_pairs/answers_gt.txt \
    --preds     QA_pairs/predictions.txt \
    --show-errors

# Build offline BM25 + FAISS indexes from corpus
python build_offline.py

# Run ablation experiments
python run_ablation.py
```

## Corpus & Indexes

| Directory | Size | Description |
|-----------|------|-------------|
| `corpus/filtered_texts_small/` | 39 MB | Small corpus for local testing |
| `corpus_index/corpus_index_small/` | 72 MB | BM25 + FAISS indexes for small corpus |
| `corpus_index/corpus_index_reference/` | 33 MB | Official reference corpus (released 3/17) |
| `corpus_index/corpus_index_big/` | 609 MB | Large corpus — excluded from repo |

Switch corpus by editing `CORPUS_INDEX_DIR` in `rag.py`.

## File Structure

```
rag.py                    # Core RAGModel class (submitted to autograder)
llm.py                    # OpenRouter LLM wrapper
build_offline.py          # Build BM25 + FAISS indexes from corpus JSON
filter_URL.py             # Filter corpus by URL patterns
filter_regex.py           # Regex-based text cleaning
filter_quality.py         # Quality-based passage filtering
evaluate_rag_model.py     # Autograder runner
eval_local.py             # Local F1/EM evaluation
run_ablation.py           # Ablation experiment runner
run.sh                    # Autograder entry point
data_mining.py            # URL fetching utility
QA_pairs/
├── questions.txt         # Input questions
├── answers_gt.txt        # Ground truth answers (|‑separated multi-ref)
├── predictions.txt       # Last model predictions
└── reference.jsonl       # Official dev set
corpus/
└── filtered_texts_small/ # Web-crawled passages (JSON)
corpus_index/
├── corpus_index_small/   # Indexes for small corpus
└── corpus_index_reference/ # Official reference indexes
```

## Autograder Constraints

- Python 3.10.12, `torch>=2.0.0`, `faiss-cpu>=1.7.4`
- Embedding model ≤ 400M params; model files ≤ 400 MB
- 100 questions must complete in ≤ 60 s total
- `OPENROUTER_API_KEY` environment variable required

## Allowed LLM Models (via OpenRouter)

- `meta-llama/llama-3.1-8b-instruct` (default)
- `qwen/qwen3-8b`
- `qwen/qwen-2.5-7b-instruct`
- `mistralai/mistral-7b-instruct`
- `allenai/olmo-3-7b-instruct`
