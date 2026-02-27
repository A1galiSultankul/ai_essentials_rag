"""
LLM-as-judge evaluation for the RAG pipeline.
Loads question-answer pairs (e.g. from the project Google Sheet), runs the RAG agent
on each question, then uses Gemini as judge to score correctness. Reports accuracy.
"""
import os
import csv
import time
import argparse
from pathlib import Path
from typing import Optional, List, Dict

# Add project root and services to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "services"))

from agent import answer_with_rag


def _call_with_retry(fn, max_retries=8, base_delay=20):
    """Call fn(), retrying on rate-limit errors with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            err_name = type(e).__name__
            err_str = str(e)
            is_rate_limit = (
                "ResourceExhausted" in err_name
                or "429" in err_str
                or "quota" in err_str.lower()
                or "rate" in err_str.lower()
                or "RESOURCE_EXHAUSTED" in err_str
            )
            if is_rate_limit and attempt < max_retries - 1:
                wait = base_delay * (2 ** attempt)
                print(f"  Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...", flush=True)
                time.sleep(wait)
            else:
                raise
    return fn()


def load_qa_pairs(csv_path: str) -> List[Dict]:
    """
    Load QA pairs from a CSV with columns: id, paper, question, answer
    (or similar: question and answer columns required).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"QA CSV not found: {csv_path}")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize keys (strip, lowercase)
            row = {k.strip().lower(): v for k, v in row.items()}
            if "question" in row and "answer" in row:
                rows.append(row)
    return rows


def llm_judge(
    question: str,
    reference_answer: str,
    model_answer: str,
    model_name: str = "gemini-2.0-flash",
) -> dict:
    """
    Use Gemini as judge: rate whether model_answer is correct given reference_answer.
    Returns dict with score (0 or 1), reason, and raw response.
    """
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required for LLM-as-judge.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    prompt = f"""You are an impartial judge for a question-answering system.

Question: {question}

Reference (ground-truth) answer: {reference_answer}

Model's answer: {model_answer}

Does the model's answer convey the same key facts and meaning as the reference, allowing for paraphrasing and slightly different wording? Answer with a single word: YES or NO. Then on a new line, in one short sentence, explain why."""

    response = model.generate_content(prompt)
    text = (response.text or "").strip().upper()
    yes = "YES" in text.split()[0] if text else False
    score = 1 if yes else 0
    return {"score": score, "reason": text, "raw": response.text}


def run_evaluation(
    qa_csv: str,
    output_csv: Optional[str] = None,
    max_samples: Optional[int] = None,
    judge_model: str = "gemini-2.0-flash",
    rag_top_k: int = 5,
    rag_score_threshold: float = 0.3,
    use_reranker: bool = False,
    retrieve_top_n: int = 20,
    rerank_top_k: int = 5,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> dict:
    """
    Run RAG on each question, then LLM-as-judge. Return metrics and optional CSV.
    """
    qa_pairs = load_qa_pairs(qa_csv)
    if max_samples is not None:
        qa_pairs = qa_pairs[:max_samples]

    results = []
    correct = 0
    for i, row in enumerate(qa_pairs):
        q = row.get("question", "").strip()
        ref = row.get("answer", "").strip()
        if not q:
            continue
        # RAG answer (with retry on rate limit)
        print(f"[{i+1}/{len(qa_pairs)}] {q[:80]}...", flush=True)
        rag_out = _call_with_retry(
            lambda _q=q: answer_with_rag(
                question=_q,
                top_k=rag_top_k,
                score_threshold=rag_score_threshold,
                model_name=judge_model,
                use_reranker=use_reranker,
                retrieve_top_n=retrieve_top_n,
                rerank_top_k=rerank_top_k,
                reranker_model=reranker_model,
            ))
        pred = rag_out.get("answer", "")
        # Judge (with retry on rate limit)
        judge_out = _call_with_retry(
            lambda _q=q, _ref=ref, _pred=pred: llm_judge(
                question=_q, reference_answer=_ref,
                model_answer=_pred, model_name=judge_model,
            ))
        score = judge_out["score"]
        correct += score
        results.append({
            "id": row.get("id", i + 1),
            "question": q[:200],
            "reference_answer": ref[:200],
            "model_answer": pred[:200],
            "score": score,
            "reason": judge_out.get("reason", "")[:300],
        })

    n = len(results)
    accuracy = (correct / n * 100) if n else 0.0
    metrics = {
        "total": n,
        "correct": correct,
        "accuracy_percent": round(accuracy, 2),
    }

    if output_csv:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "question", "reference_answer", "model_answer", "score", "reason"])
            w.writeheader()
            w.writerows(results)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG with LLM-as-judge on QA pairs")
    parser.add_argument("--qa-csv", default="data/qa_pairs.csv", help="Path to QA CSV (id, paper, question, answer)")
    parser.add_argument("--output-csv", default=None, help="Path to write per-row results CSV")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of QA pairs (for testing)")
    parser.add_argument("--judge-model", default="gemini-2.0-flash", help="Gemini model for judge")
    parser.add_argument("--rag-top-k", type=int, default=5, help="RAG top_k")
    parser.add_argument("--rag-threshold", type=float, default=0.3, help="RAG score threshold")
    parser.add_argument("--use-reranker", action="store_true", help="Enable cross-encoder reranker")
    parser.add_argument("--retrieve-top-n", type=int, default=20, help="Number of candidates to fetch before reranking")
    parser.add_argument("--rerank-top-k", type=int, default=5, help="Number of chunks to keep after reranking")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Cross-encoder model name")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: Set GEMINI_API_KEY (e.g. from https://aistudio.google.com/apikey)")
        sys.exit(1)

    metrics = run_evaluation(
        qa_csv=args.qa_csv,
        output_csv=args.output_csv,
        max_samples=args.max_samples,
        judge_model=args.judge_model,
        rag_top_k=args.rag_top_k,
        rag_score_threshold=args.rag_threshold,
        use_reranker=args.use_reranker,
        retrieve_top_n=args.retrieve_top_n,
        rerank_top_k=args.rerank_top_k,
        reranker_model=args.reranker_model,
    )
    print("Evaluation results (Retrieval + Generation, LLM-as-judge)")
    print("=" * 50)
    print(f"Total QA pairs: {metrics['total']}")
    print(f"Correct (judge YES): {metrics['correct']}")
    print(f"Accuracy: {metrics['accuracy_percent']}%")
    if args.output_csv:
        print(f"Per-row results written to: {args.output_csv}")


if __name__ == "__main__":
    main()
