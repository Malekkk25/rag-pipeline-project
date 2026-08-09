import json
from os import path
from pydoc import text
import re
from pathlib import Path
from dataclasses import dataclass,asdict
from tracemalloc import start
from typing import List
from pypdf import PdfReader


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    position: int # chunk index within the source document

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def load_documents(data_dir : str) ->List[dict]:
    """Load every .txt and .pdf file in data_dir into {source, text} dicts."""
    docs = []
    for path in sorted(Path(data_dir).glob("**/*")):
        if path.suffix.lower() == ".txt":
            docs.append({"source": path.name , "text" :read_txt(path)})
        elif path.suffix.lower() == ".pdf":
            docs.append({"source": path.name , "text" :read_pdf(path)}) 
        return docs       


#ensure no context is lost when dividing long documents into vector database embeddings.
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks by word count."""
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(words):
        piece = words[start:start + chunk_size]
        if piece:
            chunks.append(" ".join(piece))
        start += step
    return chunks


def build_chunks(data_dir:str ="data/raw" , chunk_size:int= 4000 ,overlap:int=50) -> List[str]:
    docs =load_documents(data_dir)
    if not docs:
        printf(f"No .txt or .pdf files found in {data_dir}/ - add some documents first.")
    all_chunks = []
    for doc in docs:
        pieces =chunk_text(doc["text"], chunk_size,overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append(Chunk(id=f"{doc['source']}::{i}",text=piece,source=doc["source"],position=i))
    return all_chunks

#asdict(c): A function from Python's built-in dataclasses module. Because Python's json library cannot write custom class instances directly, asdict() converts each object c into a standard Python dictionary (e.g., {"text": "...", "metadata": "..."}).
def save_chunks(chunks:List[chunk] , out_path: str ="data/chunks.json"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True) 
    with open(out_path ,"w",encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks],f,ensure_ascii=False,indent=2)

if __name__ == "__main__":
    chunks=build_chunks()
    save_chunks(chunks)
    print(f"Ingested {len(chunks)} chunks -> data/chunks.json")

