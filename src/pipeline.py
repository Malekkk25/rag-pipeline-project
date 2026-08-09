"""
Step 6: The actual end-to-end RAG pipeline - retrieve -> rerank ->
generate -> verify. Every other module is a building block; this is
where they get wired together.

Run: python -m src.pipeline
"""

from src.retrieval import HybridRetriever
from src.rerank import Reranker
from src.generate import generate_answer
from src.verify import CitationVerifier


class RAGPipeline:
    def __init__(self):
        # Loading these once at startup (not per-query) is the difference
        # between a system that takes 50ms per query vs one that takes 10s.
        print("Loading retriever...")
        self.retriever = HybridRetriever()
        print("Loading reranker...")
        self.reranker = Reranker()
        print("Loading citation verifier...")
        self.verifier = CitationVerifier()
        print("Pipeline ready.\n")

    def answer(self, query: str, retrieve_k: int = 20, rerank_top_n: int = 5) -> dict:
        # 1. Retrieve broad candidate set with hybrid search
        candidates = self.retriever.search(query, k=retrieve_k)
        
        # 2. Rerank narrow with cross-encoder for high precision
        top_chunks = self.reranker.rerank(query, candidates, top_n=rerank_top_n)
        
        # 3. Generate answer with LLM using retrieved chunks
        result = generate_answer(query, top_chunks)
        
        # 4. Verify LLM claims against source texts using NLI model
        verified = self.verifier.verify_all(result["citations"])

        return {
            "query": query,
            "answer": result["answer"],
            "retrieved_chunks": top_chunks,
            "citation_checks": verified,
            "hallucination_rate": self._hallucination_rate(verified),
        }

    @staticmethod
    def _hallucination_rate(checks) -> float:
        if not checks:
            return 0.0
        unsupported = sum(1 for c in checks if not c["supported"])
        return unsupported / len(checks)


if __name__ == "__main__":
    pipeline = RAGPipeline()

    # Query targeting your first document (rag_overview.txt)
    test_query = "What is Retrieval-Augmented Generation?"
    result = pipeline.answer(test_query)

    print("===============================")
    print(f"QUERY: {result['query']}")
    print("===============================\n")

    print("--- GENERATED ANSWER ---")
    print(result["answer"])

    print("\n--- CITATION VERIFICATION ---")
    for check in result["citation_checks"]:
        status = "✅ SUPPORTED" if check["supported"] else "❌ NOT SUPPORTED"
        print(f"{status} (score={check['entailment_score']:.2f}): {check['claim']}")

    print(f"\nHallucination rate: {result['hallucination_rate']:.0%}")

    # Gracefully close Qdrant connection on exit
    pipeline.retriever.client.close()