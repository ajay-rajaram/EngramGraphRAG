"""
HRG Retrieval Pipeline.

Implements the online retrieval phase from the HippoRAG paper:
    Step 1 — Query NER        : extract named entities from query
    Step 2 — Node mapping     : map entities to KG nodes (exact + Qdrant fallback)
    Step 3 — Build igraph     : load KuzuDB edges into igraph
    Step 4 — PPR              : personalized PageRank seeded at query nodes
    Step 5 — Passage scoring  : top-K nodes → chunk lookup → ranked episodes

Run:
    ./vevenv/bin/python -m src.HRG.hrgRetrieval
"""

from __future__ import annotations

from logger import get_logger
from db.hrg_graph import HrgGraph
from helpers.entityExtraction import EntityExtraction
from helpers.nodeMapping import NodeMapping
from helpers.passageTable import PassageTable
from helpers.pprEngine import PprEngine

logger = get_logger(__name__)


class HrgRetrieval:

    def __init__(self) -> None:
        self._graph            = HrgGraph(read_only=True)
        self._entity_extractor = EntityExtraction()
        self._node_mapper      = NodeMapping(self._graph)
        self._ppr_engine       = PprEngine(self._graph)
        self._passage_table    = PassageTable()

    # ── Step 1: Query NER ────────────────────────────────────────────────────

    def query_entityExtraction(self, query: str) -> list[str]:
        """Extract named entities from the query — entry points for graph walk."""
        entities = self._entity_extractor.query_entityExtraction(query)
        logger.info("Query NER — query='%s'  entities=%s", query[:80], entities)
        return entities

    # ── Step 2: Node mapping ─────────────────────────────────────────────────

    def map_to_seed_nodes(self, entities: list[str]) -> dict[str, float]:
        """Map query entities → {node_id: normalised_weight} for PPR reset vector."""
        seeds = self._node_mapper.map(entities)
        logger.info("Seed nodes: %s", {k: round(v, 4) for k, v in seeds.items()})
        return seeds

    # ── Steps 3-5: PPR + passage scoring ─────────────────────────────────────

    def run_ppr(self, seeds: dict[str, float]) -> list[tuple[str, float]]:
        """Steps 3+4+5 — build igraph, run PPR, score passages."""
        return self._ppr_engine.run(seeds)

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Full retrieval pipeline.

        Returns [(episode_id, score), ...] sorted by score descending, capped at top_k.
        Returns empty list if no query entities map to graph nodes.
        """
        logger.info("════ HRG RETRIEVAL  query='%s' ════", query[:80])

        # Step 1 — query NER
        logger.info("── STEP 1: Query NER ──")
        entities = self.query_entityExtraction(query)
        if not entities:
            logger.warning("  No entities extracted — returning empty")
            return []
        logger.info("  Entities: %s", entities)

        # Step 2 — map to seed nodes (logged inside NodeMapping.map())
        seeds = self.map_to_seed_nodes(entities)
        if not seeds:
            logger.warning("  No graph nodes found for entities %s — returning empty", entities)
            return []
        # Steps 3+4+5 — PPR + passage scoring (logged inside PprEngine)
        ranked = self.run_ppr(seeds)

        logger.info("════ HRG DONE  results=%d  top_k=%d ════", len(ranked), top_k)
        for ep_id, score in ranked[:top_k]:
            logger.info("  → %-30s  score=%.4f", ep_id, score)

        return ranked[:top_k]


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    hrg = HrgRetrieval()

    query = "On which day did Ravi visit Denton Farmers Market?"

    print(f"\n{'═'*60}")
    print(f"Query: {query}")
    print(f"{'═'*60}")

    # Steps 1-5: ranked episode IDs
    ranked = hrg.retrieve(query, top_k=3)

    if not ranked:
        print("  (no results)")
    else:
        # Step 6: fetch passage text
        passages = hrg._passage_table.fetch(ranked)

        print(f"\n{'─'*60}")
        print("Ranked passages with text:\n")
        for i, (ep_id, score, text) in enumerate(passages, 1):
            print(f"[{i}] {ep_id}  score={score:.4f}")
            print(f"    {text}")
            print()

    hrg._node_mapper._qdrant.close()
    hrg._graph.close()

# ./vevenv/bin/python -m src.HRG.hrgRetrieval
