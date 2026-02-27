"""
Task 3 – Retrieval Accuracy Evaluation

For each evaluation question the corresponding source PDF (paper) is known.
This script:
  1. Auto-builds a paper_id → source_file mapping via majority-vote retrieval.
  2. Retrieves chunks for every question (with / without reranker).
  3. Checks whether retrieved chunks originate from the correct document.
  4. Reports per-question hit@k and aggregate retrieval accuracy.
"""
import csv
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "services"))

from embedding_manager import Embedder
from qdrant_manager import QdrantManager
from reranker import rerank


MAPPING_CACHE = Path(__file__).resolve().parent / "data" / "paper_pdf_mapping.json"


def load_qa_pairs(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip().lower(): v for k, v in row.items()}
            if "question" in row and "answer" in row:
                rows.append(row)
    return rows


def build_paper_pdf_mapping(
    qa_pairs: List[Dict],
    embedder: Embedder,
    qdrant: QdrantManager,
    collection_name: str = "pdf_documents",
    top_k: int = 10,
    score_threshold: float = 0.2,
) -> Dict[str, str]:
    """
    For each paper_id, embed its questions, retrieve chunks, and build a
    vote matrix.  Then do a greedy 1-to-1 assignment so each PDF is used
    at most once (each paper maps to a unique PDF).
    """
    if MAPPING_CACHE.exists():
        with open(MAPPING_CACHE) as f:
            mapping = json.load(f)
        print(f"Loaded cached mapping from {MAPPING_CACHE} ({len(mapping)} papers)")
        return mapping

    papers: Dict[str, List[str]] = {}
    for row in qa_pairs:
        pid = row.get("paper", "").strip()
        q = row.get("question", "").strip()
        if pid and q:
            papers.setdefault(pid, []).append(q)

    vote_matrix: Dict[str, Counter] = {}
    print("Building paper → PDF mapping via majority-vote retrieval …")
    for pid in sorted(papers, key=lambda x: int(x)):
        votes: Counter = Counter()
        for q in papers[pid]:
            emb = embedder.embed_text(q)
            results = qdrant.search_points_with_scores(
                embedding=emb,
                collection_name=collection_name,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            for r in results:
                src = r.get("metadata", {}).get("source_file")
                if src:
                    votes[src] += 1
        vote_matrix[pid] = votes
        top3 = dict(votes.most_common(3))
        print(f"  Paper {pid:>2s} votes: {top3}")

    # Greedy 1-to-1 assignment: pick (paper, pdf) pairs with highest vote
    # count first, ensuring each PDF is assigned to only one paper.
    all_pairs = []
    for pid, votes in vote_matrix.items():
        for pdf, count in votes.items():
            all_pairs.append((count, pid, pdf))
    all_pairs.sort(reverse=True)

    mapping: Dict[str, str] = {}
    used_pdfs: set = set()
    for count, pid, pdf in all_pairs:
        if pid in mapping or pdf in used_pdfs:
            continue
        mapping[pid] = pdf
        used_pdfs.add(pdf)

    print("\nFinal 1-to-1 mapping:")
    for pid in sorted(mapping, key=lambda x: int(x)):
        votes_for = vote_matrix[pid].get(mapping[pid], 0)
        print(f"  Paper {pid:>2s} → {mapping[pid]}  (votes: {votes_for})")

    unassigned = [p for p in papers if p not in mapping]
    if unassigned:
        print(f"  Unassigned papers (no unique PDF left): {unassigned}")

    MAPPING_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPING_CACHE, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved mapping to {MAPPING_CACHE}")
    return mapping


def evaluate_retrieval(
    qa_pairs: List[Dict],
    mapping: Dict[str, str],
    embedder: Embedder,
    qdrant: QdrantManager,
    collection_name: str = "pdf_documents",
    top_k: int = 5,
    score_threshold: float = 0.3,
    use_reranker: bool = False,
    retrieve_top_n: int = 20,
    rerank_top_k: int = 5,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> Dict[str, Any]:
    """
    For each question: retrieve top_k chunks, check if any come from the
    ground-truth source PDF. Return per-question results and aggregates.
    """
    results = []
    hits = 0
    total = 0

    for i, row in enumerate(qa_pairs):
        q = row.get("question", "").strip()
        pid = row.get("paper", "").strip()
        if not q or pid not in mapping:
            continue
        correct_pdf = mapping[pid]

        effective_k = retrieve_top_n if use_reranker else top_k
        emb = embedder.embed_text(q)
        chunks = qdrant.search_points_with_scores(
            embedding=emb,
            collection_name=collection_name,
            top_k=effective_k,
            score_threshold=score_threshold,
        )

        if use_reranker and chunks:
            chunks = rerank(
                query=q,
                candidates=chunks,
                top_k=rerank_top_k,
                model_name=reranker_model,
            )

        retrieved_sources = [
            c.get("metadata", {}).get("source_file", "") for c in chunks
        ]
        hit = correct_pdf in retrieved_sources
        hits += int(hit)
        total += 1

        correct_count = sum(1 for s in retrieved_sources if s == correct_pdf)
        precision_at_k = correct_count / len(chunks) if chunks else 0.0

        results.append({
            "id": row.get("id", i + 1),
            "paper": pid,
            "correct_pdf": correct_pdf,
            "question": q[:120],
            "hit": int(hit),
            "precision_at_k": round(precision_at_k, 3),
            "retrieved_sources": retrieved_sources[:top_k],
        })

        tag = "HIT" if hit else "MISS"
        print(
            f"[{total:>2}/{len(qa_pairs)}] [{tag:4s}] paper={pid:>2s}  "
            f"prec@k={precision_at_k:.2f}  q={q[:70]}…"
        )

    accuracy = hits / total if total else 0.0
    avg_precision = (
        sum(r["precision_at_k"] for r in results) / total if total else 0.0
    )

    metrics = {
        "total": total,
        "hits": hits,
        "hit_rate": round(accuracy * 100, 2),
        "avg_precision_at_k": round(avg_precision, 4),
    }
    return {"metrics": metrics, "details": results}


def main():
    parser = argparse.ArgumentParser(
        description="Measure retrieval accuracy: does the correct source PDF appear in top-k?"
    )
    parser.add_argument(
        "--qa-csv",
        default="data/RAG Documents - Sheet1.csv",
        help="Path to QA CSV",
    )
    parser.add_argument("--collection", default="pdf_documents")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--retrieve-top-n", type=int, default=20)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--output-csv", default=None, help="Save per-question results")
    parser.add_argument("--rebuild-mapping", action="store_true", help="Force rebuild paper→PDF mapping")
    args = parser.parse_args()

    if args.rebuild_mapping and MAPPING_CACHE.exists():
        MAPPING_CACHE.unlink()
        print("Deleted cached mapping – will rebuild.")

    qa_pairs = load_qa_pairs(args.qa_csv)
    if not qa_pairs:
        print(f"No QA pairs found in {args.qa_csv}")
        sys.exit(1)

    embedder = Embedder(model_name="nomic-embed-text")
    qdrant = QdrantManager(host="localhost", port=6333)

    mapping = build_paper_pdf_mapping(
        qa_pairs, embedder, qdrant, args.collection
    )

    mode = "WITH reranker" if args.use_reranker else "WITHOUT reranker"
    print(f"\n{'=' * 60}")
    print(f"Evaluating retrieval accuracy ({mode})")
    print(f"{'=' * 60}\n")

    out = evaluate_retrieval(
        qa_pairs=qa_pairs,
        mapping=mapping,
        embedder=embedder,
        qdrant=qdrant,
        collection_name=args.collection,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        use_reranker=args.use_reranker,
        retrieve_top_n=args.retrieve_top_n,
        rerank_top_k=args.rerank_top_k,
        reranker_model=args.reranker_model,
    )

    m = out["metrics"]
    print(f"\n{'=' * 60}")
    print(f"RETRIEVAL ACCURACY RESULTS ({mode})")
    print(f"{'=' * 60}")
    print(f"Total questions     : {m['total']}")
    print(f"Hits (correct doc)  : {m['hits']}")
    print(f"Hit rate            : {m['hit_rate']}%")
    print(f"Avg precision@k     : {m['avg_precision_at_k']}")
    print(f"{'=' * 60}")

    if args.output_csv:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["id", "paper", "correct_pdf", "question", "hit", "precision_at_k"],
            )
            w.writeheader()
            for r in out["details"]:
                w.writerow({k: r[k] for k in w.fieldnames})
        print(f"Per-question results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
