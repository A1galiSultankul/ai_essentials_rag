"""
Task 4 – Interactive RAG Chatbot (Terminal Demo)

Accepts user queries, retrieves relevant chunks from Qdrant (with optional
reranker), and generates answers using Gemini in real time.

Usage:
  python chatbot.py                          # basic mode
  python chatbot.py --use-reranker           # with reranker
  python chatbot.py --top-k 3 --verbose      # show sources and scores
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "services"))

from retrieval_tool import search_relevant_chunks, format_chunks_for_context


WELCOME = """
╔══════════════════════════════════════════════════════════════╗
║              RAG Chatbot – PDF Knowledge Base                ║
║  Type your question and press Enter. Type 'quit' to exit.   ║
╚══════════════════════════════════════════════════════════════╝
"""


def generate_answer(context: str, question: str, model_name: str) -> str:
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "[ERROR] GEMINI_API_KEY not set. Export it or pass via env."
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    prompt = (
        "You are a precise assistant. Answer the question using ONLY the "
        "provided context from PDF documents. If the context does not contain "
        "enough information, say so briefly. Do not invent facts.\n\n"
        f"Context from documents:\n---\n{context}\n---\n\n"
        f"Question: {question}\n\n"
        "Answer (concise and grounded in the context):"
    )
    response = model.generate_content(prompt)
    return (response.text or "").strip()


def run_chatbot(args):
    print(WELCOME)

    reranker_label = "ON" if args.use_reranker else "OFF"
    print(f"  Model: {args.model}  |  Reranker: {reranker_label}  |  top_k: {args.top_k}")
    print(f"  Collection: {args.collection}\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("  [Retrieving relevant chunks …]")
        chunks = search_relevant_chunks(
            query=query,
            collection_name=args.collection,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
            use_reranker=args.use_reranker,
            retrieve_top_n=args.retrieve_top_n,
            rerank_top_k=args.rerank_top_k,
            reranker_model=args.reranker_model,
        )

        if not chunks:
            print("  Bot: No relevant passages found in the knowledge base.\n")
            continue

        if args.verbose:
            print(f"  Retrieved {len(chunks)} chunk(s):")
            for i, c in enumerate(chunks, 1):
                src = c.get("metadata", {}).get("source_file", "?")
                idx = c.get("metadata", {}).get("chunk_index", "?")
                score = c.get("score", 0)
                rr = c.get("rerank_score")
                rr_str = f"  rerank={rr:.3f}" if rr is not None else ""
                print(f"    [{i}] {src} chunk#{idx}  score={score:.4f}{rr_str}")

        context = format_chunks_for_context(chunks, max_chars=6000)

        print("  [Generating answer …]")
        answer = generate_answer(context, query, args.model)
        print(f"\n  Bot: {answer}\n")

        if args.verbose:
            sources = list(
                dict.fromkeys(
                    c.get("metadata", {}).get("source_file", "?") for c in chunks
                )
            )
            print(f"  Sources: {', '.join(sources)}\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive RAG Chatbot")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Gemini model name")
    parser.add_argument("--collection", default="pdf_documents")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--retrieve-top-n", type=int, default=20)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show sources and scores")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY not set. Set it for answer generation.")
        print("  export GEMINI_API_KEY='your-key-here'\n")

    run_chatbot(args)


if __name__ == "__main__":
    main()
