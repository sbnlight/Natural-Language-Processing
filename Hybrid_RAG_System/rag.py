"""
Hybrid Retrieval = BM25 + FAISS -> RRF Fusion -> Cross-Encoder Rerank -> Few-Shot LLM
Constraints: 2 CPUs, 3GB RAM, no GPU
"""

import json
import re
import time
import pickle
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

# ─── Config ──────────────────────────────────────────────────────────────────

CORPUS_INDEX_DIR       = Path("corpus_index/corpus_index_small")  # Choose the corpus index you want

# Recall phase parameters
BM25_CANDIDATES  = 30    # Number of BM25 candidates
DENSE_CANDIDATES = 30    # Number of dense candidates
USE_DENSE        = True
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"  # A small model ~80MB

# Fusion and reranking phase parameters (RRF + Reranker)
RRF_K            = 60    # RRF smoothing constant (standard setting is 60)
USE_RERANKER     = True  # Whether to enable reranking (set to False if CPU overloads and times out)
RERANK_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2" # Lightweight cross-encoder
RERANK_TOP_N     = 10    # Take top 10 from RRF for reranking (do not set too large to prevent CPU timeout)
FINAL_TOP_K      = 8     # Number of passages finally returned to the LLM

# LLM parameters
LLM_TIMEOUT      = 999.0   # Shorten timeout duration, fail fast
MAX_WORKERS      = 4     # Considering CPU load and API limits, choose 4 as relatively safe concurrency number
FALLBACK_ANS     = "UNKNOWN" # Guessing UNKNOWN has at least a 5% chance of being correct

# ─── Tokenizer ───────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase + remove punctuation + split by space, for BM25."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()

# ─── RAGModel ─────────────────────────────────────────────────────────────────

class RAGModel:
    """
    Public interface (required by evaluate_rag_model.py):
        model = RAGModel()
        answers = model.predict(questions)   # List[str] -> List[str]
    """

    def __init__(self):
        t0 = time.perf_counter()
        
        # Load text data in seconds
        print("[RAGModel] Loading offline corpus.json...")
        with open(CORPUS_INDEX_DIR / "corpus.json", "r", encoding="utf-8") as f:
            corpus_data = json.load(f)
            self.passages = corpus_data["passages"]
            self.urls = corpus_data["urls"]
            
        # Load BM25 in seconds
        print("[RAGModel] Loading offline BM25 index...")
        with open(CORPUS_INDEX_DIR / "bm25_index.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
            
        # Load FAISS index in seconds
        print("[RAGModel] Loading offline FAISS index...")
        import faiss
        self.faiss_index = faiss.read_index(str(CORPUS_INDEX_DIR / "faiss_index.bin"))
        
        # Load the lightweight Embedder model itself (only used to compute Query vectors)
        from sentence_transformers import SentenceTransformer
        print(f"[RAGModel] Loading Embedder: {EMBEDDING_MODEL}...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # Load lightweight Reranker (if True)
        if USE_RERANKER:
            from sentence_transformers import CrossEncoder
            print(f"[RAGModel] Loading Reranker: {RERANK_MODEL}...")
            self.reranker = CrossEncoder(RERANK_MODEL, max_length=512, device="cpu")
        else:
            self.reranker = None
        
        print(f"[RAGModel] Ready! All offline assets loaded in {time.perf_counter()-t0:.2f}s")

    # ── 1. Retrieval (Recall, Fusion, Reranking) ─────────────────────────────────────────────

    def _retrieve_bm25(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """BM25 retrieval, returns [(passage_index, raw_score), ...]"""
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def _retrieve_dense(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Dense retrieval, returns[(passage_index, cosine_score), ...]"""
        if self.faiss_index is None or self.embedder is None:
            return[]
        q_vec = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        scores, indices = self.faiss_index.search(q_vec, top_k)
        return[(int(indices[0][i]), float(scores[0][i])) for i in range(len(indices[0]))]

    def _retrieve(self, query: str) -> List[str]:
        """
        Hybrid retrieval process:
        ① Recall separately: BM25 takes Top-30, FAISS takes Top-30
        ② RRF fusion: use reciprocal rank to solve the issue of two scores not being on the same scale
        ③ Reranking: use Cross-Encoder to re-score and sort the Top-10 from RRF
        """
        # --- Step 1: Dual-path recall ---
        bm25_results  = self._retrieve_bm25(query, BM25_CANDIDATES)
        dense_results = self._retrieve_dense(query, DENSE_CANDIDATES) if USE_DENSE else[]

        # --- Step 2: RRF fusion ---
        rrf_scores = {}
        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (idx, _) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        # Sort by RRF score, truncate Top-N for the reranker
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_n_indices = [idx for idx, score in sorted_rrf[:RERANK_TOP_N]]

        # --- Step 3: Cross-Encoder reranking ---
        if USE_RERANKER and self.reranker is not None:
            # Assemble (query, passage) pairs for the model to score
            pairs = [[query, self.passages[idx]] for idx in top_n_indices]
            rerank_scores = self.reranker.predict(pairs)
            
            # Bind scores with original idx and re-sort
            reranked_results = sorted(zip(top_n_indices, rerank_scores), key=lambda x: x[1], reverse=True)
            final_indices =[idx for idx, _ in reranked_results[:FINAL_TOP_K]]
        else:
            final_indices = top_n_indices[:FINAL_TOP_K]
    
        return[self.passages[i] for i in final_indices]

    # ── 2. Generate answer ────────────────────────────────────────────────────────────

    def _build_prompt(self, question: str, passages: List[str]) -> str:
        """Use Few-Shot Prompt to force LLM to only output entity words, improving F1 score"""
        context = "\n---\n".join(passages)
        return (
            "Answer the question using ONLY the context below. "
            "Reply with the absolute shortest possible answer (under 10 words, preferably just a named entity, number, or date). "
            "No explanation. No full sentences. No punctuation at the end. "
            "If the answer cannot be found, make your best guess based on the context.\n\n"
            "Example 1:\n"
            "Context: Dan Klein is a Professor in the EECS department.\n"
            "Question: What is Dan Klein's title?\n"
            "Answer: Professor\n\n"
            "Example 2:\n"
            "Context: The CS288 course requires CS188 as prerequisite.\n"
            "Question: What is the prerequisite for CS288?\n"
            "Answer: CS188\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    def _call_llm_safe(self, prompt: str) -> str:
        """Call llm.py with timeout protection, return FALLBACK_ANS on any exception."""
        try:
            from llm import call_llm  # autograder will overwrite llm.py, import here
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(call_llm, prompt)
                result = future.result(timeout=LLM_TIMEOUT)
            if not isinstance(result, str):
                return FALLBACK_ANS
            answer = result.strip().replace("\n", " ")
            return answer if answer else FALLBACK_ANS
        except (FuturesTimeout, Exception) as e:
            # print error
            print(f"[DEBUG ERROR] {type(e).__name__}: {e}")
            return FALLBACK_ANS

    def _normalize_answer(self, text: str) -> str:
        """Normalize common formats to improve EM match rate."""
        # Remove trailing punctuation
        text = text.strip().rstrip(".,;:")
        # Remove leading articles (a strong optimization for F1 metric)
        text = re.sub(r'^(the|a|an)\s+', '', text, flags=re.IGNORECASE)
        return text.strip()

    def _answer_one(self, question: str) -> str:
        passages = self._retrieve(question)
        prompt   = self._build_prompt(question, passages)
        answer   = self._normalize_answer(self._call_llm_safe(prompt))
        return answer.replace("\n", " ").strip() or FALLBACK_ANS

    # ── 3. Public interface ────────────────────────────────────────────────────────────

    def predict(self, questions: List[str]) -> List[str]:
        """
        Main entry point. 100 questions must be completed within 60 seconds.
        """
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            answers = list(executor.map(self._answer_one, questions))
        return answers