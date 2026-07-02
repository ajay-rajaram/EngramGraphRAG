# EngramGraphRAG

> **Graph-native RAG powered by KuzuDB, igraph PPR, and local LLMs **

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![KuzuDB](https://img.shields.io/badge/graph-KuzuDB-orange.svg)](https://kuzudb.com/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-green.svg)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Neuroscience Behind It

Your brain doesn't retrieve memories by scanning every thought you've ever had.  
It retrieves by **association** — one concept activates connected concepts, which activate others, until the right memory surfaces.

This is what the **hippocampus** does. It is the brain's indexing system:

```
                    ┌─────────────────────────────────┐
                    │         HIPPOCAMPUS              │
                    │                                  │
   New experience → │  stores as connected engrams     │ → retrievable memory
                    │  (entities linked by relations)  │
                    │                                  │
   Query/cue      → │  spreads activation through      │ → recalled memory
                    │  the association network         │
                    └─────────────────────────────────┘
```

An **engram** is the neuroscience term for a single stored memory trace — a specific pattern of connected neurons that represents a fact or event.

**EngramGraphRAG mirrors this exactly:**

| Brain | EngramGraphRAG |
|---|---|
| Hippocampus | KuzuDB knowledge graph |
| Engrams (memory traces) | HrgNode entities + HrgRelation edges |
| Associative activation | Personalized PageRank graph walk |
| Memory consolidation | Synonym edges (semantically linked nodes) |
| Episodic memory | Passage chunks (HrgChunk) |
| Memory retrieval by cue | Query NER → seed nodes → PPR |

When you ask a question, EngramGraphRAG seeds the graph at your query entities and lets activation **spread through the knowledge graph** — exactly how the hippocampus retrieves a memory by association, not by brute-force search.

> *"Retrieval is not search. It is remembering."*

---

## What is EngramGraphRAG?

EngramGraphRAG is a **biologically-inspired, knowledge-graph RAG system** that retrieves relevant passages using **Personalized PageRank** over a **KuzuDB property graph** — not cosine similarity over flat vectors.

Inspired by [HippoRAG (NeurIPS 2024)](https://arxiv.org/abs/2405.14831), rebuilt with:
- **KuzuDB** instead of NetworkX — persistent, queryable via Cypher, visualisable
- **Predicate-frequency weighted PPR** — our key innovation over the original paper
- **all-minilm local embeddings** — synonym edges via Qdrant, CPU only

---

## How it works

```
INDEXING  (run once per document)
──────────────────────────────────────────────────────────────
  Document
    -> Step 1: Named Entity Recognition     [LLM]
    -> Step 2: OpenIE Triple Extraction     [LLM]
    -> Step 3: Build Knowledge Graph        [KuzuDB]
    -> Step 4: Synonym Edges                [Qdrant + all-minilm]

RETRIEVAL  (per query, 2 LLM calls total)
──────────────────────────────────────────────────────────────
  Query
    -> Step 1: Query NER                   [LLM]
    -> Step 2: Map entities to graph nodes [KuzuDB exact + Qdrant fallback]
    -> Step 3: Build weighted igraph       [KuzuDB Cypher]
               predicate weight = 1 / frequency  <-- our innovation
    -> Step 4: Personalized PageRank       [igraph, CPU, milliseconds]
    -> Step 5: Score passages              [pure Python]
    -> Step 6: Fetch passage text          [PostgreSQL]
    -> Step 7: RAG QA answer               [LLM]
```

### Why predicate-frequency weighting?

Standard HippoRAG treats all graph edges equally. We weight edges by inverse predicate frequency:

| Predicate | Frequency | Weight | Effect |
|---|---|---|---|
| `"attended boarding school in"` | 1 | **1.000** | PPR flows strongly |
| `"born in"` | 3 | **0.333** | Moderate flow |
| `"located in"` | 6 | **0.167** | Weak flow — suppressed |

Specific facts surface above generic relationships. The answer node rises to the top.

---

## Accuracy

Tested on 30 queries across factual, relational, temporal, and multi-hop inference types:

| Approach | Accuracy |
|---|---|
| HippoRAG (unweighted PPR) | 88.9% |
| **EngramGraphRAG (weighted PPR)** | **92.6%** |

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | |
| [Ollama](https://ollama.com/) | latest | local LLM runtime |
| PostgreSQL | 14+ | passage text store |
| KuzuDB | 0.11+ | auto-installed via pip |

Pull the required Ollama models:
```bash
ollama pull qwen2.5:7b-instruct   # NER + OpenIE + RAG QA
ollama pull all-minilm:latest     # synonym edge embeddings (384-dim)
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/EngramGraphRAG.git
cd EngramGraphRAG

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create PostgreSQL database
psql -U postgres -c "CREATE DATABASE engramgraphrag;"

# 4. Configure — copy the template and fill in your credentials
cp config.example.toml config.toml
nano config.toml   # set your DB user/password + Ollama model names

# 5. Index a document
python examples/index_document.py path/to/your.pdf

# 6. Query
python examples/query.py "What is the main argument of this document?"
```

---

## Project Structure

```
EngramGraphRAG/
  config.toml              # Ollama models, DB, graph, Qdrant settings
  requirements.txt
  logger.py                # session logger -> logs/logfiles/

  hrgIndexing.py           # Indexing pipeline entry point
  hrgRetrieval.py          # Retrieval steps 1-5 (returns episode IDs + scores)
  hrgAnswering.py          # Full pipeline: retrieve -> fetch -> answer

  db/
    hrg_graph.py           # All KuzuDB operations (HrgNode, HrgChunk, HrgRelation...)

  helpers/
    entityExtraction.py    # Step 1: passage NER + query NER
    tripletExtraction.py   # Step 2: OpenIE triple extraction
    buildKG.py             # Step 3: write to KuzuDB
    synonymEdges.py        # Step 4: Qdrant embed + synonym edges (batch)
    nodeMapping.py         # Retrieval Step 2: entity -> graph node
    pprEngine.py           # Retrieval Steps 3-5: weighted igraph PPR
    passageTable.py        # Passage text store (PostgreSQL)

  llm/
    client.py              # LLMService — Ollama wrapper
    schemas.py             # JSON output schemas
    prompts/
      passage_ner.py       # Passage NER prompt
      query_ner.py         # Query NER prompt
      oie.py               # OpenIE triple extraction prompt
      rag_qa.py            # RAG QA prompt (HippoRAG paper)

  ingest/
    file_processor.py      # Load PDF / DOCX / TXT
    cleaner.py             # Strip headers, references, page numbers
    chunker.py             # Token-aware sentence-boundary chunker

  examples/
    index_document.py      # How to index a file
    query.py               # How to run a query

  data/                    # KuzuDB + Qdrant stores (gitignored)
  logs/                    # Session logs (gitignored)
```

---

## KuzuDB Graph Schema

```
HrgNode       (id STRING PK, text STRING)          -- named entities
HrgChunk      (id STRING PK)                       -- one per passage
HrgRelation   (HrgNode -> HrgNode,                 -- OpenIE triples
               predicate STRING, chunk_id STRING)
HrgSynonym    (HrgNode -> HrgNode,                 -- semantic bridges
               cosine_score DOUBLE)
HrgChunkNode  (HrgChunk -> HrgNode)                -- passage-node matrix
```

View your graph live with [Kuzu Explorer](https://docs.kuzudb.com/visualization/):
```bash
docker run -p 8000:8000 \
  -v $(pwd)/data/graph:/database \
  -e KUZU_FILE=database.kz \
  kuzudb/explorer:latest
```

---

## Key Design Decisions

| Decision | Why |
|---|---|
| KuzuDB over NetworkX | Persistent, Cypher-queryable, Docker-viewable, production-grade |
| igraph over NetworkX PPR | C-backed, 10-100x faster for graph algorithms |
| Predicate-frequency weights | Specific facts surface above generic hub relationships |
| Episode dedup guard | Zero redundant LLM calls on re-indexing |
| Local embeddings (all-minilm) | CPU-only, 15ms per node, fully airgapped |
| Full passages to LLM | Best accuracy — sentence extraction experiments showed regressions |

---

## LLM Cost per Operation

| Operation | LLM calls |
|---|---|
| Index 1 chunk (800 tokens) | 2 (NER + OpenIE) |
| Query | 2 (Query NER + RAG QA) |
| Synonym edges (batch) | 0 (local embeddings) |
| All other steps | 0 |

---


## License

MIT — see [LICENSE](LICENSE)

---

## Citation

If you use EngramGraphRAG in your work, please also cite the original HippoRAG paper:

```bibtex
@inproceedings{gutierrez2024hipporag,
  title     = {HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models},
  author    = {Gutierrez, Bernal Jimenez and Shu, Yiheng and Gu, Yu and Kamigaito, Hidetaka and Su, Yu},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```
