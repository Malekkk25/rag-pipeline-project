"""
Step 3: Hybrid retrieval - dense (embeddings) + sparse (BM25),
combined with Reciprocal Rank Fusion (RRF).

Why hybrid: dense embeddings are great at semantic similarity but can
miss exact keyword matches (e.g. "Article 17" or a product SKU). BM25
is great at exact matches but blind to paraphrasing. Combining both
and fusing the rankings gets you the best of both instead of picking
one and living with its blind spot.

Uses Qdrant in local/on-disk mode - no server to run, it's just a
library here, but it's the same client API you'd use against a real
Qdrant deployment later.

Run: python -m src.retrieval
"""

import json
from typing import List, Dict

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class HybridRetriever:
    def __init__(
        self,
        chunks_path: str = "data/chunks.json",
        embed_model: str = "all-MiniLM-L6-v2",
        qdrant_path: str = "data/qdrant_store",
    ):
        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = json.load(f)  # list of {id, text, source, position}
        self.id_to_chunk: Dict[str, dict] = {c["id"]: c for c in self.chunks}

        # --- sparse index (BM25) ---
        tokenized = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

        # --- dense index (Qdrant, on-disk) ---
        self.embedder = SentenceTransformer(embed_model)
        self.client = QdrantClient(path=qdrant_path)
        self.collection = "chunks"
        
        # Check if collection exists
        if not self.client.collection_exists(self.collection):
            self.build_dense_index()

    def build_dense_index(self):
        vectors = self.embedder.encode([c["text"] for c in self.chunks], show_progress_bar=True)
        
        # Fixed API: Use collection_exists and create_collection instead of recreate_collection
        if self.client.collection_exists(collection_name=self.collection):
            self.client.delete_collection(collection_name=self.collection)
            
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
        )
        
        points = [
            PointStruct(id=i, vector=vectors[i].tolist(), payload={"chunk_id": self.chunks[i]["id"]}) 
            for i, c in enumerate(self.chunks)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def _dense_search(self, query: str, k: int) -> List[str]:
        qvec = self.embedder.encode([query])[0].tolist()
        
        # Fixed API: Use query_points instead of search
        response = self.client.query_points(
            collection_name=self.collection, 
            query=qvec, 
            limit=k
        )
        
        return [h.payload["chunk_id"] for h in response.points]

    def _sparse_search(self, query: str, k: int) -> List[str]:
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i]["id"] for i in ranked]

    def search(self, query: str, k: int = 2, rrf_k: int = 60) -> List[dict]:
        """Run both retrievers and fuse rankings with Reciprocal Rank Fusion."""
        dense_ids = self._dense_search(query, k)
        sparse_ids = self._sparse_search(query, k)

        scores: Dict[str, float] = {}
        for rank, cid in enumerate(dense_ids):
            scores[cid] = scores.get(cid, 0) + 1 / (rrf_k + 1 + rank)
        for rank, cid in enumerate(sparse_ids):
            scores[cid] = scores.get(cid, 0) + 1 / (rrf_k + 1 + rank)
            
        ranked_ids = sorted(scores, key=scores.get, reverse=True)[:k]
        return [self.id_to_chunk[cid] for cid in ranked_ids]


if __name__ == "__main__":
    retriever = HybridRetriever()
    test_query = "What is Retrieval-Augmented Generation?"
    
    print(f"\n--- Top 5 Search Results for: '{test_query}' ---\n")
    for r in retriever.search(test_query, k=5):
        print(f"ID: {r['id']}")
        print(f"Text preview: {r['text'][:80]}...")
        print("-" * 50)