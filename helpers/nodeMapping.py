"""
Step 2 — Node Mapping (retrieval path).

Maps query entities from Step 1 to HrgNode IDs in KuzuDB.
Two-track lookup per entity:
    1. Exact match   — get_node_by_text() (normalisation-aware)
    2. Qdrant fallback — embed entity → search hrg_nodes collection (cosine >= 0.75)

After mapping, computes node specificity (1 / number_of_chunks_containing_node)
and returns a normalised weight dict ready for the PPR reset vector.

Run:
    ./vevenv/bin/python -m src.HRG.helpers.nodeMapping
"""

from __future__ import annotations

import ollama
from qdrant_client import QdrantClient

from logger import get_logger
from db.hrg_graph import HrgGraph
from helpers.synonymEdges import _COLLECTION, _EMBED_MODEL, _QDRANT_PATH

logger = get_logger(__name__)

_FALLBACK_THRESHOLD = 0.75
_FALLBACK_TOP_K     = 1


class NodeMapping:

    def __init__(self, graph: HrgGraph, qdrant_path: str = _QDRANT_PATH) -> None:
        self._graph        = graph
        self._qdrant       = QdrantClient(path=qdrant_path)
        self._qdrant_ready = self._check_collection()

    def _check_collection(self) -> bool:
        existing = {c.name for c in self._qdrant.get_collections().collections}
        if _COLLECTION not in existing:
            logger.warning(
                "NodeMapping: Qdrant collection '%s' not found — "
                "Qdrant fallback disabled. Run synonymEdgeCreation() first.",
                _COLLECTION,
            )
            return False
        return True

    # ── Public ───────────────────────────────────────────────────────────────

    def map(self, entities: list[str]) -> dict[str, float]:
        """
        Map query entities to seed node IDs with normalised specificity weights.

        Returns {node_id: weight} where weights sum to 1.0.
        Returns {} if no entity maps to any graph node.
        """
        logger.info("── STEP 2: Node Mapping ── entities=%s", entities)
        seed_raw: dict[str, float] = {}

        for entity in entities:
            nid = self._exact_match(entity) or self._qdrant_fallback(entity)

            if nid is None:
                logger.info("  MISS  '%s' — not found in graph or Qdrant", entity)
                continue

            chunks      = self._graph.get_chunks_for_nodes([nid])
            n_chunks    = max(len(chunks.get(nid, [])), 1)
            specificity = 1.0 / n_chunks
            seed_raw[nid] = specificity

            logger.info(
                "  HIT   '%s' → %s  appears_in=%d chunks  raw_specificity=%.4f",
                entity, nid, n_chunks, specificity,
            )

        if not seed_raw:
            logger.warning("  No entities mapped to graph nodes — PPR cannot run")
            return {}

        total    = sum(seed_raw.values())
        result   = {nid: w / total for nid, w in seed_raw.items()}
        logger.info("  SEEDS (normalised weights):")
        for nid, w in result.items():
            logger.info("    %s  →  weight=%.4f", nid, w)
        return result

    # ── Private ──────────────────────────────────────────────────────────────

    def _exact_match(self, entity: str) -> str | None:
        nid = self._graph.get_node_by_text(entity)
        if nid:
            logger.info("    match=EXACT   '%s' → %s", entity, nid)
        return nid

    def _qdrant_fallback(self, entity: str) -> str | None:
        if not self._qdrant_ready:
            return None
        vector = ollama.embeddings(model=_EMBED_MODEL, prompt=entity)["embedding"]
        result = self._qdrant.query_points(
            collection_name=_COLLECTION,
            query=vector,
            limit=_FALLBACK_TOP_K,
            score_threshold=_FALLBACK_THRESHOLD,
            with_payload=True,
        )
        if not result.points:
            logger.info("    match=QDRANT  '%s' → no match above %.2f threshold", entity, _FALLBACK_THRESHOLD)
            return None
        nid   = result.points[0].payload["node_id"]
        score = result.points[0].score
        logger.info("    match=QDRANT  '%s' → %s  cosine=%.3f", entity, nid, score)
        return nid


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    graph  = HrgGraph(read_only=True)
    mapper = NodeMapping(graph)

    test_entities = [
        "Green Valley School",
        "Sweet Crumbs",
        "Ravi",
        "FreshFarm Suppliers",
        "nonexistent entity xyz",
    ]

    print("Entity → seed node mapping\n")
    seeds = mapper.map(test_entities)
    for nid, weight in seeds.items():
        print(f"  {nid:<45} weight={weight:.4f}")

    mapper._qdrant.close()
    graph.close()

# ./vevenv/bin/python -m src.HRG.helpers.nodeMapping
