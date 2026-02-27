"""
Task 2 – Chunking Configuration Experiment

Tests multiple chunk_size / chunk_overlap combinations, re-populates Qdrant
for each, and measures retrieval accuracy (hit rate and precision@k).
No Gemini API calls required – only Ollama embeddings + Qdrant.

Usage:
  python chunking_experiment.py                   # full grid (slow)
  python chunking_experiment.py --quick            # 3 configs for a quick test
  python chunking_experiment.py --output-csv results/chunking.csv
"""
import sys
import csv
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "services"))

from processing import pdf2chunks
from embedding_manager import Embedder
from qdrant_manager import QdrantManager
from reranker import rerank

DATA_FOLDER = Path(__file__).resolve().parent / "data"
QA_CSV = DATA_FOLDER / "RAG Documents - Sheet1.csv"
COLLECTION_PREFIX = "chunk_exp"

FULL_GRID: List[Tuple[int, int]] = [
    (500, 50),
    (500, 100),
    (750, 100),
    (750, 200),
    (1000, 100),
    (1000, 200),
    (1000, 300),
    (1500, 200),
    (1500, 300),
    (2000, 300),
]

QUICK_GRID: List[Tuple[int, int]] = [
    (500, 100),
    (1000, 200),
    (1500, 300),
]


def load_qa_pairs(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip().lower(): v for k, v in row.items()}
            if "question" in row and "paper" in row:
                rows.append(row)
    return rows


def populate_collection(
    embedder: Embedder,
    qdrant: QdrantManager,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    """Re-chunk PDFs, embed, and insert into a fresh Qdrant collection."""
    try:
        qdrant.delete_collection(collection_name)
    except Exception:
        pass

    qdrant.create_collection(name=collection_name, vector_size=768)
    pdf_files = sorted(DATA_FOLDER.glob("*.pdf"))
    total = 0

    for fi, pdf_file in enumerate(pdf_files, 1):
        chunks_dict = pdf2chunks(pdf_file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunks_dict.get(pdf_file.name, [])
        print(f"    [{fi}/{len(pdf_files)}] {pdf_file.name}: {len(chunks)} chunks", flush=True)
        for idx, chunk in enumerate(chunks):
            embedding = embedder.embed_text(chunk)
            metadata = {
                "source_file": pdf_file.name,
                "chunk_index": idx,
                "chunk_size": len(chunk),
            }
            qdrant.insert_point(
                embedding=embedding,
                collection_name=collection_name,
                chunk_text=chunk,
                metadata=metadata,
            )
            total += 1

    print(f"  Populated {collection_name}: {total} chunks from {len(pdf_files)} PDFs", flush=True)
    return total


def build_mapping_for_collection(
    qa_pairs: List[Dict],
    embedder: Embedder,
    qdrant: QdrantManager,
    collection_name: str,
) -> Dict[str, str]:
    """Auto-map paper_id → source_file via majority-vote retrieval."""
    papers: Dict[str, List[str]] = {}
    for row in qa_pairs:
        pid = row.get("paper", "").strip()
        q = row.get("question", "").strip()
        if pid and q:
            papers.setdefault(pid, []).append(q)

    mapping: Dict[str, str] = {}
    for pid in sorted(papers, key=lambda x: int(x)):
        votes: Counter = Counter()
        for q in papers[pid]:
            emb = embedder.embed_text(q)
            results = qdrant.search_points_with_scores(
                embedding=emb,
                collection_name=collection_name,
                top_k=10,
                score_threshold=0.2,
            )
            for r in results:
                src = r.get("metadata", {}).get("source_file")
                if src:
                    votes[src] += 1
        if votes:
            mapping[pid] = votes.most_common(1)[0][0]
    return mapping


def measure_retrieval_accuracy(
    qa_pairs: List[Dict],
    mapping: Dict[str, str],
    embedder: Embedder,
    qdrant: QdrantManager,
    collection_name: str,
    top_k: int = 5,
    score_threshold: float = 0.3,
    use_reranker: bool = False,
    retrieve_top_n: int = 20,
    rerank_top_k: int = 5,
) -> Dict[str, Any]:
    hits = 0
    total = 0
    prec_sum = 0.0

    for row in qa_pairs:
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
            chunks = rerank(query=q, candidates=chunks, top_k=rerank_top_k)

        sources = [c.get("metadata", {}).get("source_file", "") for c in chunks]
        hit = correct_pdf in sources
        hits += int(hit)
        correct_count = sum(1 for s in sources if s == correct_pdf)
        prec_sum += correct_count / len(chunks) if chunks else 0.0
        total += 1

    acc = hits / total * 100 if total else 0.0
    avg_prec = prec_sum / total if total else 0.0
    return {
        "total": total,
        "hits": hits,
        "hit_rate": round(acc, 2),
        "avg_precision_at_k": round(avg_prec, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Chunking configuration experiment")
    parser.add_argument("--quick", action="store_true", help="Run only 3 key configs")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--retrieve-top-n", type=int, default=20)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    args = parser.parse_args()

    grid = QUICK_GRID if args.quick else FULL_GRID
    qa_pairs = load_qa_pairs(str(QA_CSV))
    if not qa_pairs:
        print(f"No QA pairs in {QA_CSV}")
        sys.exit(1)

    embedder = Embedder(model_name="nomic-embed-text")
    qdrant = QdrantManager(host="localhost", port=6333)

    all_results: List[Dict[str, Any]] = []

    for chunk_size, chunk_overlap in grid:
        col = f"{COLLECTION_PREFIX}_{chunk_size}_{chunk_overlap}"
        print(f"\n{'=' * 60}")
        print(f"Config: chunk_size={chunk_size}, overlap={chunk_overlap}")
        print(f"{'=' * 60}")

        t0 = time.time()
        n_chunks = populate_collection(embedder, qdrant, col, chunk_size, chunk_overlap)
        pop_time = time.time() - t0
        print(f"  Populate time: {pop_time:.1f}s")

        mapping = build_mapping_for_collection(qa_pairs, embedder, qdrant, col)

        t1 = time.time()
        mode_label = "reranker" if args.use_reranker else "no-reranker"
        metrics = measure_retrieval_accuracy(
            qa_pairs, mapping, embedder, qdrant, col,
            top_k=args.top_k,
            use_reranker=args.use_reranker,
            retrieve_top_n=args.retrieve_top_n,
            rerank_top_k=args.rerank_top_k,
        )
        eval_time = time.time() - t1

        row = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "num_chunks": n_chunks,
            "mode": mode_label,
            **metrics,
            "populate_sec": round(pop_time, 1),
            "eval_sec": round(eval_time, 1),
        }
        all_results.append(row)

        print(f"  Hit rate: {metrics['hit_rate']}%  Avg prec@k: {metrics['avg_precision_at_k']}")
        print(f"  Eval time: {eval_time:.1f}s")

        try:
            qdrant.delete_collection(col)
        except Exception:
            pass

    print(f"\n{'=' * 70}")
    print("CHUNKING EXPERIMENT RESULTS")
    print(f"{'=' * 70}")
    header = f"{'Size':>6} {'Overlap':>7} {'Chunks':>7} {'HitRate':>8} {'AvgP@k':>8}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['chunk_size']:>6} {r['chunk_overlap']:>7} {r['num_chunks']:>7} "
            f"{r['hit_rate']:>7.1f}% {r['avg_precision_at_k']:>8.4f}"
        )

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            w.writerows(all_results)
        print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
