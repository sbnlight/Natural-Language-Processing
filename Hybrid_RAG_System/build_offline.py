#!/usr/bin/env python3
"""
build_offline.py — Offline RAG index building
Converts raw scraped JSON/JSONL into lightweight artifacts for Gradescope (no GPU, 3GB RAM).
"""

import json
import re
import pickle
from pathlib import Path
from typing import List
import numpy as np

# Import necessary building libraries
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer

# ─── Configuration Parameters ───
CORPUS_DIR = Path("./corpus/filtered_texts_big")
OUTPUT_DIR = Path("./corpus_index/corpus_index_big")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Must be exactly identical to _tokenize in rag.py!
def _tokenize(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    passages = []
    urls =[]

    print(f"Step 1: Reading raw JSON/JSONL files from {CORPUS_DIR}...")
    
    # Support both .json and .jsonl files
    json_files = list(CORPUS_DIR.glob("*.json")) + list(CORPUS_DIR.glob("*.jsonl"))
    
    if not json_files:
        raise FileNotFoundError(f"No json/jsonl files found in {CORPUS_DIR}!")

    # 1. Parse all text data
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                
                # Check if it's a JSONL file (by extension or by trying to read line by line)
                if file_path.suffix == '.jsonl':
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)  # Parse single line
                        url = data.get("url", "")
                        text = data.get("text", "").strip() # Directly get "text"
                        if text:
                            passages.append(text)
                            urls.append(url)
                            
                else:
                    # Fallback for original standard JSON format with "chunks"
                    data = json.load(f)
                    url = data.get("url", "")
                    
                    # Handle if the old json actually directly has "text" instead of "chunks"
                    if "text" in data:
                        text = data.get("text", "").strip()
                        if text:
                            passages.append(text)
                            urls.append(url)
                    else:
                        for chunk in data.get("chunks",[]):
                            text = chunk.get("text", "").strip()
                            if text:
                                passages.append(text)
                                urls.append(url)
                                
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    print(f"Successfully loaded {len(passages)} passages.")
    
    if len(passages) == 0:
        print("Error: No text passages were extracted. Please check your JSON format!")
        return

    # 2. Save pure text structure (massively compress size for direct loading in rag.py)
    print("\nStep 2: Saving lightweight corpus.json...")
    corpus_data = {
        "passages": passages,
        "urls": urls
    }
    with open(OUTPUT_DIR / "corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus_data, f, ensure_ascii=False)
    print(f"-> Saved to {OUTPUT_DIR / 'corpus.json'}")

    # 3. Build and save BM25 index
    print("\nStep 3: Building BM25 index...")
    tokenized_corpus = [_tokenize(p) for p in passages]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(OUTPUT_DIR / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print(f"-> Saved BM25 index to {OUTPUT_DIR / 'bm25_index.pkl'}")

    # 4. Build and save FAISS vector index
    print(f"\nStep 4: Building FAISS dense index using {EMBEDDING_MODEL}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    
    # Batch compute embeddings (run slowly locally with GPU or no time limit)
    embeddings = embedder.encode(
        passages,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalization to make dot product equal to cosine similarity
    )
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim) # Inner Product index
    faiss_index.add(embeddings)
    
    faiss.write_index(faiss_index, str(OUTPUT_DIR / "faiss_index.bin"))
    print(f"-> Saved FAISS index (dim={dim}, vectors={faiss_index.ntotal}) to {OUTPUT_DIR / 'faiss_index.bin'}")
    
    print("\nOffline build complete!")

if __name__ == "__main__":
    main()