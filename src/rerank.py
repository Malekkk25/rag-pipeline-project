"""
Step 4: Cross-encoder reranking.

Dense/sparse retrieval score the query and each document INDEPENDENTLY
(that's what makes them fast enough to search thousands of chunks). A
cross-encoder looks at the query and a document TOGETHER in one forward
pass, which is far more accurate but much slower - so we only run it on
the top candidates from hybrid search, not the whole corpus. This
"retrieve broad, rerank narrow" pattern is standard in production RAG.

Run: python -m src.rerank
"""

from typing import List
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[dict], top_n:int =6) -> List[dict]:
        """Rerank candidates by relevance to the query."""
        pairs = [[query, c["text"]] for c in chunks]
        scores = self.model.predict(pairs)
        for c ,s in zip(chunks, scores):
            c["rerank_score"] = float(s)
        return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)[:top_n]

if __name__ == "__main__":
    from src.retrieval import HybridRetriever

    retriever = HybridRetriever()
    reranker = Reranker()

    test_query = "What is Retrieval-Augmented Generation?"
    candidates = retriever.search(test_query, k=20)
    top_results = reranker.rerank(test_query, candidates, top_n=5)

    for r in top_results:
        print(f"{r['rerank_score']:.3f}  {r['id']}  -  {r['text'][:80]}...")