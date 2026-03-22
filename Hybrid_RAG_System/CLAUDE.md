# CS288 Assignment 3 — RAG QA System

## Project Overview
UC Berkeley CS288 (Natural Language Processing) homework 3. Build a Retrieval-Augmented Generation (RAG) system to answer questions about the EECS Berkeley website. Runs on Gradescope autograder with strict constraints: 2 CPUs, 3 GB RAM, no GPU, 100 questions within 60 seconds.

## Team Division
- **Data Engineer (同学 A):** Web crawling, corpus cleaning, chunking, official corpus comparison
- **RAG Architect (同学 B, this user):** Pipeline, indexes, `run.sh`, LLM integration, performance tuning
- **QA & PM (同学 C):** QA dataset construction (100+ pairs), prompt engineering, ablation experiments, report

## Key Files
| File | Purpose |
|------|---------|
| `rag.py` | Core `RAGModel` class — the only file submitted to autograder |
| `llm.py` | LLM wrapper via OpenRouter (autograder overwrites this) |
| `run.sh` | Autograder entry: `bash run.sh <questions> <predictions>` |
| `evaluate_rag_model.py` | Autograder runner: loads `RAGModel`, calls `predict()`, writes output |
| `eval_local.py` | Local F1/EM evaluation; supports `|`-separated multi-reference answers |
| `build_offline.py` | Builds offline BM25 + FAISS indexes |
| `run_ablation.py` | Ablation experiment runner |

## RAG Pipeline (rag.py)
1. **Recall:** BM25 (sparse) + FAISS (dense) each return Top-30 candidates
2. **Fusion:** Reciprocal Rank Fusion (RRF, k=60) merges the two ranked lists
3. **Reranking:** Cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rescores Top-10 from RRF
4. **Generation:** Few-shot prompt → OpenRouter LLM (default: `meta-llama/llama-3.1-8b-instruct`) → answer normalized (strip articles, trailing punctuation)

## Corpus Indexes
Located in `corpus_index/`:
- `corpus_index_small/` — small corpus for fast local testing
- `corpus_index_big/` — large corpus (~20k pages) for full runs
- `corpus_index_reference/` — official reference corpus released 3/17

Each index folder contains: `corpus.json` (passages + URLs), `bm25_index.pkl`, `faiss_index.bin`.

Active index is selected by `CORPUS_INDEX_DIR` constant in `rag.py`.

## QA Data
Located in `QA_pairs/`:
- `questions.txt` — one question per line
- `answers_gt.txt` — one answer per line (multi-answer: `|`-separated)
- `predictions.txt` — model output from last run
- `reference.jsonl` — official dev set

## LLM Setup
- API: OpenRouter (`OPENROUTER_API_KEY` env var required)
- Allowed models: `meta-llama/llama-3.1-8b-instruct`, `qwen/qwen3-8b`, `qwen/qwen-2.5-7b-instruct`, etc.
- `max_tokens=64`, `temperature=0.0` for deterministic short answers
- Autograder **overwrites** `llm.py` — do not rely on any custom changes to it surviving submission

## Running Locally
```bash
# Run full pipeline (produces predictions.txt)
bash run.sh QA_pairs/questions.txt QA_pairs/predictions.txt

# Evaluate predictions
python eval_local.py --questions QA_pairs/questions.txt \
                     --answers   QA_pairs/answers_gt.txt \
                     --preds     QA_pairs/predictions.txt \
                     --show-errors
```

## Autograder Constraints
- Python 3.10.12, `torch>=2.0.0`, `faiss-cpu>=1.7.4`
- Embedding model ≤ 400M params, model files ≤ 400MB
- 100 questions must complete in ≤ 60 s total (0.6 s/question average)
- `MAX_WORKERS=4` threads for concurrent LLM calls
- `FALLBACK_ANS = "UNKNOWN"` on timeout/error

## Metrics
- **Exact Match (EM):** after lowercasing, removing punctuation and articles
- **F1:** token overlap between prediction and reference
- `eval_local.py` replicates the official `evaluate-v1.1.py` scoring logic exactly
