"""
Step 5b: Citation verification - the "quality layer" almost every RAG
demo skips.

After the model generates an answer with citations, we check each
(claim, cited_source) pair with a Natural Language Inference (NLI)
model: does the source chunk actually ENTAIL the claim, or is the
model citing a source that sounds related but doesn't really back up
what it said?

Run: python -m src.verify
"""

from typing import List, Dict
import numpy as np
from sentence_transformers import CrossEncoder

# This checkpoint outputs logits in this label order - check the model
# card if you swap models, the label order isn't universal.
LABELS = ["contradiction", "entailment", "neutral"]


class CitationVerifier:
    def __init__(
        self, 
        model_name: str = "cross-encoder/nli-deberta-v3-base", 
        entailment_threshold: float = 0.5
    ):
        self.model = CrossEncoder(model_name)
        self.threshold = entailment_threshold  # Fixed typo: thereshold -> threshold

    def verify(self, claim: str, source_text: str) -> Dict:
        # Fixed: Pass a list containing a tuple [(source_text, claim)]
        raw_scores = self.model.predict([(source_text, claim)])[0]
        probs = self._softmax(raw_scores)
        entailment_score = float(probs[LABELS.index("entailment")])
        
        return {
            "claim": claim,
            "entailment_score": entailment_score,
            "supported": entailment_score >= self.threshold,
        }

    def verify_all(self, citations: List[Dict]) -> List[Dict]:
        """citations: list of {claim, source_text}"""
        return [self.verify(c["claim"], c["source_text"]) for c in citations]

    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()


if __name__ == "__main__":
    verifier = CitationVerifier()

    # Test cases built using content from the RAG overview file
    test_cases = [
        # Case 1: Supported claim (Entailment)
        {
            "claim": "RAG combines an information retrieval system with a generative language model.",
            "source_text": "Retrieval-Augmented Generation (RAG) is an architectural approach in artificial intelligence that combines an information retrieval system with a generative large language model.",
        },
        # Case 2: Unsupported / Contradicted claim
        {
            "claim": "RAG models do not require any external knowledge or document collections.",
            "source_text": "Retrieval-Augmented Generation (RAG) relies on retrieving relevant text chunks from external document collections to ground the LLM's response.",
        },
    ]

    print("\n--- Running Citation Verification ---\n")
    for result in verifier.verify_all(test_cases):
        status = "SUPPORTED" if result["supported"] else "NOT SUPPORTED"
        print(f"[{status}] (score={result['entailment_score']:.3f})")
        print(f"Claim: {result['claim']}\n")