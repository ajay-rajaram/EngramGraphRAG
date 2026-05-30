"""
Example: Index a PDF document into EngramGraphRAG.

Run from the EngramGraphRAG directory:
    python examples/index_document.py path/to/document.pdf
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hrgIndexing import HrgIndexing
from ingest.chunker import ChunkText
from ingest.cleaner import CleanText
from ingest.file_processor import FileProcessor


def episode_id(filename: str, idx: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", Path(filename).stem.lower())[:30]
    return f"ep_{slug}_{idx:04d}"


def index(path: str) -> int:
    processor = FileProcessor()
    cleaner   = CleanText()
    chunker   = ChunkText()
    hrg       = HrgIndexing()

    total = 0
    for raw_text, metadata in processor.process(path):
        chunks = chunker.chunk(cleaner.clean(raw_text))
        print(f"{metadata['filename']} -> {len(chunks)} chunks")
        for idx, chunk in enumerate(chunks):
            hrg.process(chunk, episode_id=episode_id(metadata["filename"], idx))
            total += 1

    print("Running synonym edge creation (batch step)...")
    hrg.synonymEdgeCreation()
    print(f"Done — {total} chunks indexed")
    return total


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/index_document.py <path_to_file_or_folder>")
        sys.exit(1)
    index(sys.argv[1])
