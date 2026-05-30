"""
HippoRAG Indexing Pipeline.

Implements the offline indexing phase from the HippoRAG paper:
    Step 1 — Passage NER         : extract named entities  → helpers/entityExtraction.py
    Step 2 — OpenIE triples      : extract (subject, predicate, object)
    Step 3 — Build KG            : write entities + edges to KuzuDB
    Step 4 — Synonymy edges      : Qdrant entity collection + synonym edges
    Step 5 — Passage-node matrix : node → episode_id tracking

Run:
    ./vevenv/bin/python -m src.HRG.hrgIndexing
"""

from logger import get_logger
from db.hrg_graph import HrgGraph
from helpers.buildKG import BuildKG
from helpers.entityExtraction import EntityExtraction
from helpers.passageTable import PassageTable
from helpers.synonymEdges import SynonymEdges
from helpers.tripletExtraction import TripletExtraction

logger = get_logger(__name__)


class HrgIndexing:

    def __init__(self):
        self._entity_extractor  = EntityExtraction()
        self._triplet_extractor = TripletExtraction()
        self._graph             = HrgGraph()
        self._kg_builder        = BuildKG(self._graph)
        self._synonym_edges     = SynonymEdges(self._graph)
        self._passage_table     = PassageTable()

    # ── Step 1: Named Entity Recognition ─────────────────────────────────────

    def entityExtraction(self, text: str) -> list[str]:
        return self._entity_extractor.passage_entityExtraction(text)

    # ── Step 2: OpenIE Triple Extraction ─────────────────────────────────────

    def OI_tripletExtraction(self, text: str, entities: list[str]) -> list[list[str]]:
        return self._triplet_extractor.tripletExtraction(text, entities)

    # ── Step 3: Build Knowledge Graph ────────────────────────────────────────

    def buildKG(
        self,
        entities: list[str],
        triples: list[list[str]],
        episode_id: str,
    ) -> dict[str, int]:
        """Write entities, relation edges, and chunk-node links to KuzuDB."""
        return self._kg_builder.build(entities, triples, episode_id)

    # ── Step 4: Synonymy Edges ────────────────────────────────────────────────

    def synonymEdgeCreation(self) -> int:
        """
        Batch step — run after all passages are indexed.

        Embeds every HrgNode, upserts to Qdrant hrg_nodes, adds HrgSynonym
        edges for pairs with cosine > threshold (default 0.8).
        Returns the number of synonym edges written.
        """
        return self._synonym_edges.run()

    # ── Step 5: Passage-Node Matrix ───────────────────────────────────────────

    def passageNodeMatrixCreation(self, entities: list[str], episode_id: str) -> None:
        """No-op — HrgChunkNode edges written in buildKG already encode the passage-node matrix."""
        _ = entities, episode_id

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def process(self, text: str, episode_id: str = "") -> dict:
        # Guard first — skip all LLM work if this episode is already indexed
        if episode_id and self._graph.chunk_exists(episode_id):
            logger.warning(
                "process SKIP [episode_dedup]: '%s' already indexed — "
                "no LLM calls made (0 entities, 0 triples)",
                episode_id,
            )
            return {"entities": [], "triples": [], "kg_stats": {}, "skipped": True}

        entities = self.entityExtraction(text)
        logger.info("Extracted %d entities: %s", len(entities), entities)

        triples = self.OI_tripletExtraction(text, entities)
        logger.info("Extracted %d triples: %s", len(triples), triples)

        kg_stats: dict[str, int] = {}
        if episode_id:
            kg_stats = self.buildKG(entities, triples, episode_id)
            logger.info("KG written: %s", kg_stats)
            self._passage_table.store(episode_id, text)
            logger.info("Passage stored: %s", episode_id)

        return {"entities": entities, "triples": triples, "kg_stats": kg_stats}


if __name__ == "__main__":
    hrg = HrgIndexing()
    text = (
    """A few days later, Ravi realized that the increased cookie orders from Green Valley School were using flour much faster than expected. To avoid future shortages, he signed a monthly supply agreement with FreshFarm Suppliers for automatic flour deliveries every Tuesday. Carlos coordinated the recurring delivery schedule and informed Maya that each shipment would now include flour, sugar, and butter together. Meanwhile, Priya from Green Valley School asked Ravi whether Sweet Crumbs could also provide strawberry cakes for the school's monthly student celebrations. Ravi discussed the request with Linda at Denton Farmers Market, and Linda agreed to reserve extra strawberries every weekend for the bakery. Ravi then asked Arjun to prepare the delivery van with insulated storage boxes so the strawberry cakes could be transported safely to the school without melting or getting damaged.
"""
    )
    # Steps 1-3: NER + OpenIE + KG build (per passage)
    result = hrg.process(text, episode_id="ep_test002")
    print("Entities:", result["entities"])
    print(f"\nTriples ({len(result['triples'])}):")
    for t in result["triples"]:
        print(f"  {t[0]}  --{t[1]}-->  {t[2]}")
    print(f"\nKG stats: {result['kg_stats']}")
    g = hrg._graph
    print(f"DB nodes={g.node_count}  relations={g.relation_count}  chunks={g.chunk_count}")

    # Step 4: Synonymy edges — run once after all passages are indexed
    print("\nRunning synonymEdgeCreation...")
    added = hrg.synonymEdgeCreation()
    print(f"Synonym edges added: {added}  (total synonyms: {g.synonym_count})")

# ./vevenv/bin/python -m src.HRG.hrgIndexing
