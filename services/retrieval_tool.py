"""
Retrieval tool: embeds a query and returns top-k relevant chunks from Qdrant.
Used by the RAG agent to ground answers in the PDF corpus.
"""
from typing import List, Dict, Any

from embedding_manager import Embedder
from qdrant_manager import QdrantManager


def search_relevant_chunks(
    query: str,
    collection_name: str = "pdf_documents",
    top_k: int = 5,
    score_threshold: float = 0.3,
    host: str = "localhost",
    port: int = 6333,
    model_name: str = "nomic-embed-text",
) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant text chunks from Qdrant for a given query.

    Args:
        query: User question or search text
        collection_name: Qdrant collection name
        top_k: Number of chunks to return
        score_threshold: Minimum similarity score (0–1)
        host: Qdrant host
        port: Qdrant port
        model_name: Ollama embedding model name

    Returns:
        List of dicts with keys: text, score, metadata (source_file, chunk_index, etc.)
    """
    embedder = Embedder(model_name=model_name)
    qdrant = QdrantManager(host=host, port=port)
    query_embedding = embedder.embed_text(query)
    results = qdrant.search_points_with_scores(
        embedding=query_embedding,
        collection_name=collection_name,
        top_k=top_k,
        score_threshold=score_threshold,
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
