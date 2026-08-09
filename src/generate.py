"""
Step 5 (generation half): ask Gemini to answer using ONLY the
retrieved+reranked chunks, citing which numbered source each claim
comes from, e.g. "...fines up to 4% of global turnover [2]." The
citation verification module then checks each of these claims
against the real chunk text.

Requires GEMINI_API_KEY to be set in your environment (see
.env.example). Get a free key at https://aistudio.google.com/apikey
- no credit card needed for the free tier.
"""

import os
import re
from typing import List, Dict
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

SYSTEM_PROMPT = """You answer questions using ONLY the numbered sources given to you.
Rules:
- Every factual sentence must end with a citation like [1] or [2] pointing to the source it came from.
- If the sources don't contain the answer, say so plainly - do not use outside knowledge.
- Only cite multiple sources in one sentence if the sentence genuinely draws on both.
"""


def format_sources(chunks: List[dict]) -> str:
    return "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks))


def generate_answer(query: str, chunks: List[dict], model: str = "gemini-3.6-flash") -> Dict:
    sources_block = format_sources(chunks)
    user_prompt = f"Sources:\n{sources_block}\n\nQuestion: {query}"

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=800,
        ),
    )
    answer_text = response.text

    # Pull out (claim, cited_chunk) pairs so verify.py can check each one.
    citations = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer_text):
        for match in re.finditer(r"\[(\d+)]", sentence):
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(chunks):
                claim = re.sub(r"\[\d+]", "", sentence).strip()
                citations.append({
                    "claim": claim,
                    "source_text": chunks[idx]["text"],
                    "source_id": chunks[idx]["id"],
                })

    return {"answer": answer_text, "citations": citations}