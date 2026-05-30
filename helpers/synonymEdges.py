"""
Step 4 — Synonymy Edges.

Batch step run after all passages are indexed. For every HrgNode in KuzuDB:
    1. Embed the node text with all-minilm (384-dim)
    2. Upsert the vector into the Qdrant `hrg_nodes` collection
    3. Search for similar nodes (cosine > threshold, default 0.8)
    4. Write HrgSynonym edges in KuzuDB for qualifying pairs

This is run once per indexing batch — NOT per passage. Calling it multiple times
is safe: HrgGraph.add_synonym() skips edges that already exist, and Qdrant upserts
are idempotent.

Storage:
    KuzuDB  — HrgSynonym edges (persistent)
    Qdrant  — hrg_nodes collection at data/hrg_qdrant/ (persistent, isolated from
               VortexEO's main data/qdrant/)

Run:
    ./vevenv/bin/python -m src.HRG.helpers.synonymEdges
"""

from __future__ import annotations

import hashlib
import uuid

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from logger import get_logger
from db.hrg_graph import HrgGraph

logger = get_logger(__name__)

_COLLECTION  = "hrg_nodes"
_VECTOR_SIZE = 384
_EMBED_MODEL = "all-minilm:latest"
_QDRANT_PATH = "data/hrg_store/qdrant"


def _qdrant_id(node_id: str) -> str:
    """Stable UUID for Qdrant derived from the node_id string."""
    return str(uuid.UUID(bytes=hashlib.md5(node_id.encode()).digest()))


class SynonymEdges:

    def __init__(
        self,
        graph: HrgGraph,
        qdrant_path: str = _QDRANT_PATH,
        threshold: float = 0.8,
        top_k: int = 10,
    ) -> None:
        self._graph     = graph
        self._threshold = threshold
        self._top_k     = top_k
        self._qdrant    = QdrantClient(path=qdrant_path)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._qdrant.get_collections().collections}
        if _COLLECTION not in existing:
            self._qdrant.create_collection(
                collection_name=_COLLECTION,
                vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
            )
            logger.info("SynonymEdges: created Qdrant collection '%s'", _COLLECTION)

    def _embed(self, text: str) -> list[float]:
        return ollama.embeddings(model=_EMBED_MODEL, prompt=text)["embedding"]

    def run(self) -> int:
        """
        Embed all HrgNodes → upsert to Qdrant → add synonym edges for similar pairs.

        Returns the number of new HrgSynonym edges written to KuzuDB.
        """
        nodes = self._graph.get_all_nodes()   # {node_id: text}
        if not nodes:
            logger.info("SynonymEdges: no nodes to process")
            return 0

        logger.info("SynonymEdges: embedding %d nodes", len(nodes))

        # Single embedding pass — cache vectors in memory
        vectors: dict[str, list[float]] = {
            nid: self._embed(text) for nid, text in nodes.items()
        }

        # Upsert all vectors into Qdrant (idempotent)
        points = [
            PointStruct(
                id=_qdrant_id(nid),
                vector=vec,
                payload={"node_id": nid},
            )
            for nid, vec in vectors.items()
        ]
        self._qdrant.upsert(collection_name=_COLLECTION, points=points)
        logger.info("SynonymEdges: upserted %d points to Qdrant", len(points))

        # Search for similar neighbours + write synonym edges
        edges_added = 0
        for nid, vec in vectors.items():
            result = self._qdrant.query_points(
                collection_name=_COLLECTION,
                query=vec,
                limit=self._top_k + 1,      # +1 to skip self
                score_threshold=self._threshold,
                with_payload=True,
            )
            for hit in result.points:
                neighbour_id = hit.payload["node_id"]
                if neighbour_id == nid:
                    continue
                self._graph.add_synonym(nid, neighbour_id, float(hit.score))
                edges_added += 1

        logger.info(
            "SynonymEdges: %d nodes → %d synonym edges (threshold=%.2f)",
            len(nodes), edges_added, self._threshold,
        )
        return edges_added


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import shutil
    from db.hrg_graph import HrgGraph
    from helpers.buildKG import BuildKG

    KG_PATH     = "data/hrg_syn_test"
    QDRANT_PATH = "data/hrg_syn_qdrant_test"
    shutil.rmtree(KG_PATH,     ignore_errors=True)
    shutil.rmtree(QDRANT_PATH, ignore_errors=True)

    graph   = HrgGraph(KG_PATH)
    builder = BuildKG(graph)

    # Seed two passages with overlapping concepts
    builder.build(
        entities=["Prometheus", "Node Exporter", "Grafana"],
        triples=[
            ["Prometheus", "scrapes", "Node Exporter"],
            ["Grafana", "visualises", "Prometheus"],
        ],
        episode_id="ep_001",
    )
    builder.build(
        entities=["Prometheus monitoring", "Grafana dashboard", "Blackbox Exporter"],
        triples=[
            ["Prometheus monitoring", "exposes", "metrics"],
            ["Grafana dashboard", "shows", "Prometheus monitoring"],
        ],
        episode_id="ep_002",
    )

    print(f"Nodes before synonymy : {graph.node_count}")
    print(f"Synonyms before       : {graph.synonym_count}")

    syn = SynonymEdges(graph, qdrant_path=QDRANT_PATH, threshold=0.7)
    added = syn.run()

    print(f"Synonym edges added   : {added}")
    print(f"Synonyms after        : {graph.synonym_count}")

    syn._qdrant.close()
    graph.clear()
    graph.close()
    shutil.rmtree(KG_PATH,     ignore_errors=True)
    shutil.rmtree(QDRANT_PATH, ignore_errors=True)
    print("Done.")

# ./vevenv/bin/python -m src.HRG.helpers.synonymEdges
