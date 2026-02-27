"""
RAG agent: uses a retrieval tool (Qdrant) and Gemini to answer questions
grounded in the PDF corpus. Supports both simple RAG (retrieve then generate)
and optional Gemini function-calling flow.
"""
import os
from typing import List, Dict, Any, Optional

from retrieval_tool import search_relevant_chunks, format_chunks_for_context


DEFAULT_MODEL = "gemini-2.0-flash"


def _get_gemini_model(model_name: Optional[str] = None):
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Set GEMINI_API_KEY in the environment. "
            "Get a free key at https://aistudio.google.com/apikey"
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name or DEFAULT_MODEL)


def answer_with_rag(
    question: str,
    top_k: int = 5,
    score_threshold: float = 0.3,
    model_name: Optional[str] = None,
    collection_name: str = "pdf_documents",
    max_context_chars: int = 6000,
    use_reranker: bool = False,
    retrieve_top_n: int = 20,
    rerank_top_k: int = 5,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> Dict[str, Any]:
    """
    Answer a question using retrieval (tool) + Gemini generation.

    1. Retrieve relevant chunks from Qdrant (the "tool").
    2. Format them as context and send to Gemini with the question.
    3. Return the model answer plus metadata (sources, scores).

    Args:
        question: User question
        top_k: Number of chunks to retrieve
        score_threshold: Minimum similarity for chunks
        model_name: Gemini model (default: gemini-1.5-flash)
        collection_name: Qdrant collection
        max_context_chars: Max characters of context to pass to the LLM

    Returns:
        Dict with keys: answer, context_used, sources (list of source_file), scores
    """
    # 1) Tool: retrieve relevant chunks
    results = search_relevant_chunks(
        query=question,
        collection_name=collection_name,
        top_k=top_k,
        score_threshold=score_threshold,
        use_reranker=use_reranker,
        retrieve_top_n=retrieve_top_n,
        rerank_top_k=rerank_top_k,
        reranker_model=reranker_model,
    )
    context = format_chunks_for_context(results, max_chars=max_context_chars)
    sources = [
        r.get("metadata", {}).get("source_file")
        for r in results
        if r.get("metadata", {}).get("source_file")
    ]
    scores = [r.get("score") for r in results if "score" in r]

    # 2) Generate answer with Gemini
    model = _get_gemini_model(model_name)
    prompt = f"""You are a precise assistant. Answer the question using ONLY the provided context from PDF documents. If the context does not contain enough information, say so briefly. Do not invent facts.

Context from documents:
---
{context}
---

Question: {question}

Answer (concise and grounded in the context):"""
    response = model.generate_content(prompt)
    answer = response.text.strip() if response.text else ""

    return {
        "answer": answer,
        "context_used": context,
        "sources": sources,
        "scores": scores,
        "num_chunks": len(results),
    }


def answer_with_agent_tool(
    question: str,
    top_k: int = 5,
    score_threshold: float = 0.3,
    model_name: Optional[str] = None,
    collection_name: str = "pdf_documents",
    use_reranker: bool = False,
    retrieve_top_n: int = 20,
    rerank_top_k: int = 5,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> str:
    """
    Simple entry point: run RAG and return only the answer string.
    """
    out = answer_with_rag(
        question=question,
        top_k=top_k,
        score_threshold=score_threshold,
        model_name=model_name,
        collection_name=collection_name,
        use_reranker=use_reranker,
        retrieve_top_n=retrieve_top_n,
        rerank_top_k=rerank_top_k,
        reranker_model=reranker_model,
    )
    return out["answer"]
