"""
Retrieval tool: embeds a query and returns top-k relevant chunks from Qdrant,
optionally reranking a larger candidate set with a cross-encoder.
"""
from typing import List, Dict, Any

from embedding_manager import Embedder
from qdrant_manager import QdrantManager
from reranker import rerank


def search_relevant_chunks(
    query: str,
    collection_name: str = "pdf_documents",
    top_k: int = 5,
    score_threshold: float = 0.3,
    host: str = "localhost",
    port: int = 6333,
    model_name: str = "nomic-embed-text",
    # Reranker controls
    use_reranker: bool = False,
    retrieve_top_n: int = 20,
    rerank_top_k: int = 5,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant text chunks from Qdrant for a given query.

    Args:
        query: User question or search text
        collection_name: Qdrant collection name
        top_k: Number of chunks to return (used when reranker is off)
        score_threshold: Minimum similarity score (0–1)
        host: Qdrant host
        port: Qdrant port
        model_name: Ollama embedding model name
        use_reranker: whether to rerank a larger candidate pool
        retrieve_top_n: number of candidates to fetch from Qdrant when reranking
        rerank_top_k: number of results to keep after reranking
        reranker_model: cross-encoder model name

    Returns:
        List of dicts with keys: text, score, metadata (source_file, chunk_index, etc.)
        If reranking, also includes rerank_score.
    """
    embedder = Embedder(model_name=model_name)
    qdrant = QdrantManager(host=host, port=port)
    query_embedding = embedder.embed_text(query)

    effective_top_k = retrieve_top_n if use_reranker else top_k

    results = qdrant.search_points_with_scores(
        embedding=query_embedding,
        collection_name=collection_name,
        top_k=effective_top_k,
        score_threshold=score_threshold,
    )

    if use_reranker and results:
        results = rerank(
            query=query,
            candidates=results,
            top_k=rerank_top_k,
            model_name=reranker_model,
        )

    return results


def format_chunks_for_context(results: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    """
    Format retrieved chunks into a single context string for the LLM.

    Args:
        results: Output from search_relevant_chunks
        max_chars: Approximate maximum context length (may truncate later chunks)

    Returns:
        Single string with numbered chunks, suitable for prompt context.
    """
    parts = []
    total = 0
    for i, r in enumerate(results, 1):
        src = r.get("metadata", {}).get("source_file", "?")
        snippet = (r.get("text") or "")[:2000].strip()
        block = f"[{i}] (source: {src})\n{snippet}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts) if parts else "No relevant passages found."
