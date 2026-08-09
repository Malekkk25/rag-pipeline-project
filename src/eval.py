"""
Step 7: Evaluation harness - the difference between "I built a RAG
system" and "I built a RAG system and here's how well it performs."

Measures:
- Retrieval quality: precision@k, recall@k, MRR against a labeled
  QA set (each question has known relevant chunk ids).
- Hallucination rate: fraction of citations flagged as unsupported
  by the verification layer, averaged across the test set.

Run: python -m src.eval
"""

import json
from pathlib import Path
from typing import List, Dict
from src.pipeline import RAGPipeline


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant hit, 0 if none."""
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1 / rank
    return 0.0


def run_eval(qa_path: str = "data/eval_set.json", k: int = 5) -> Dict:
    path = Path(qa_path)
    if not path.exists():
        # Fallback check for alternative folder naming
        alt_path = Path("eval/qa_pairs.json")
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Evaluation dataset not found at {qa_path} or {alt_path}")

    with open(path, encoding="utf-8") as f:
        qa_pairs = json.load(f)

    pipeline = RAGPipeline()
    precisions, recalls, mrrs, halluc_rates = [], [], [], []

    print(f"\n--- Running Evaluation on {len(qa_pairs)} Questions ---\n")

    for item in qa_pairs:
        print(f"Evaluating: '{item['question']}'")
        result = pipeline.answer(item["question"])
        retrieved_ids = [c["id"] for c in result["retrieved_chunks"]]

        precisions.append(precision_at_k(retrieved_ids, item["relevant_chunk_ids"], k))
        recalls.append(recall_at_k(retrieved_ids, item["relevant_chunk_ids"], k))
        mrrs.append(mrr(retrieved_ids, item["relevant_chunk_ids"]))
        halluc_rates.append(result["hallucination_rate"])

    # Gracefully close vector store connection
    pipeline.retriever.client.close()

    n = len(qa_pairs)
    return {
        f"precision@{k}": sum(precisions) / n,
        f"recall@{k}": sum(recalls) / n,
        "mrr": sum(mrrs) / n,
        "avg_hallucination_rate": sum(halluc_rates) / n,
        "n_questions": n,
    }


if __name__ == "__main__":
    metrics = run_eval()
    
    print("\n===============================")
    print("      SUMMARY EVAL RESULTS     ")
    print("===============================")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name:<22}: {value:.2%}" if "rate" in name or "@" in name else f"{name:<22}: {value:.3f}")
        else:
            print(f"{name:<22}: {value}")