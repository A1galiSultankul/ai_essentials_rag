"""
Cross-encoder reranker for query + candidate chunks.
Uses sentence-transformers cross-encoder models to score (query, text) pairs.
"""
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder


_model_cache = {}


def _get_model(model_name: str) -> CrossEncoder:
    """
    Lazy-load and cache the cross-encoder model to avoid repeated downloads.
    """
    if model_name not in _model_cache:
        _model_cache[model_name] = CrossEncoder(model_name, trust_remote_code=True)
    return _model_cache[model_name]


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[Dict[str, Any]]:
    """
    Rerank candidates for a query using a cross-encoder.

    Args:
        query: user question
        candidates: list of dicts with at least "text"
        top_k: number of results to keep
        model_name: cross-encoder name

    Returns:
        candidates sorted by rerank score (descending), truncated to top_k.
    """
    if not candidates:
        return []

    model = _get_model(model_name)
    pairs = [(query, c.get("text", "")) for c in candidates]
    scores = model.predict(pairs).tolist()

    # attach rerank score, sort, slice
    for c, s in zip(candidates, scores):
        c["rerank_score"] = s

    reranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
    return reranked[: top_k or len(reranked)]
