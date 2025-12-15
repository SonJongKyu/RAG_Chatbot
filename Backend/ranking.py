# --------------------------------------------------
# File: ranking.py
# Description: Dense(Vector) 검색 + Sparse(BM25) 검색 결합 하이브리드 리랭커
# --------------------------------------------------

from rank_bm25 import BM25Okapi
import numpy as np


def hybrid_rank(query: str, faiss_results: list, embedder,
                w_dense=0.6, w_sparse=0.4):

    if not faiss_results:
        return []

    corpus = [r.get("text", "") for r in faiss_results]

    # CSV-only 문서 (text 없음)
    if all((t is None or t.strip() == "") for t in corpus):
        return faiss_results

    # BM25 준비
    tokenized = [c.split() for c in corpus]

    if any(len(toks) == 0 for toks in tokenized):
        return faiss_results

    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.split())
    if np.max(bm25_scores) > 0:
        bm25_scores = bm25_scores / np.max(bm25_scores)

    dense_scores = np.array([r["score"] for r in faiss_results])
    if np.max(dense_scores) > 0:
        dense_scores = dense_scores / np.max(dense_scores)

    final = w_dense * dense_scores + w_sparse * bm25_scores

    ranked = sorted(
        zip(faiss_results, final),
        key=lambda x: x[1],
        reverse=True
    )
    return [r[0] for r in ranked]
