import re
from pathlib import Path
from ingest.chunker import ChunkText
from ingest.cleaner import CleanText
from ingest.file_processor import FileProcessor
from hrgIndexing import HrgIndexing
from hrgAnswering import HrgAnswering

PATH = "/home/ajay/Desktop/AjayWorkSpace/VortexEO/papers/The_Life_of_Elena_Vasquez.pdf"

def episode_id(filename, idx):
    slug = re.sub(r"[^a-z0-9]+", "_", Path(filename).stem.lower())[:30]
    return f"ep_{slug}_{idx:04d}"

# ── Index ──────────────────────────────────────────────────────────────────────
processor = FileProcessor()
cleaner   = CleanText()
chunker   = ChunkText()
hrg       = HrgIndexing()

for raw_text, metadata in processor.process(PATH):
    chunks = chunker.chunk(cleaner.clean(raw_text))
    print(f"{metadata['filename']} → {len(chunks)} chunks")
    for idx, chunk in enumerate(chunks):
        ep_id = episode_id(metadata["filename"], idx)
        hrg.process(chunk, episode_id=ep_id)

hrg.synonymEdgeCreation()
print("Indexing done")

# ── Query ──────────────────────────────────────────────────────────────────────
query  = "What is GraphRAG?"
answer = HrgAnswering().ask(query)
print(f"Q: {query}")
print(f"A: {answer}")

# ./vevenv/bin/python -m src.HRG.testENT