"""
Step 3 — Build Knowledge Graph.

Takes the output of Steps 1 + 2 (entities list + triples list) for a single
passage and writes them to KuzuDB via HrgGraph:
    - Upsert every entity as an HrgNode
    - Create an HrgChunk for the passage (episode_id)
    - Create HrgRelation edges for every (subj, pred, obj) triple
    - Create HrgChunkNode links: chunk → every entity node

Run:
    ./vevenv/bin/python -m src.HRG.helpers.buildKG
"""

from __future__ import annotations

from logger import get_logger
from db.hrg_graph import HrgGraph

logger = get_logger(__name__)


class BuildKG:

    def __init__(self, graph: HrgGraph) -> None:
        self._graph = graph

    def build(
        self,
        entities: list[str],
        triples: list[list[str]],
        episode_id: str,
    ) -> dict[str, int]:
        """
        Write one passage worth of HRG data into KuzuDB.

        Returns a small stats dict so the caller can log progress.
        """
        g = self._graph

        # ── Episode-level dedup ───────────────────────────────────────────────
        if g.chunk_exists(episode_id):
            logger.warning(
                "buildKG SKIP [episode_dedup]: episode '%s' already in graph — "
                "skipping re-index (pass a unique episode_id to index a new passage)",
                episode_id,
            )
            return {"nodes": 0, "relations": 0, "skipped": True}

        # ── Chunk node ────────────────────────────────────────────────────────
        g.add_chunk(episode_id)

        # ── Entity nodes + chunk-node links ───────────────────────────────────
        entity_node_ids: set[str] = set()
        for ent in entities:
            nid = g.upsert_node(ent)
            entity_node_ids.add(nid)
            g.link_chunk_to_node(episode_id, nid)

        # ── Relation edges ────────────────────────────────────────────────────
        relations_added = 0
        for triple in triples:
            if len(triple) != 3:
                continue
            subj, pred, obj = triple
            g.add_relation(subj, pred, obj, episode_id)

            # Ensure subject + object are also linked to this chunk
            for node_text in (subj, obj):
                nid = g.upsert_node(node_text)
                if nid not in entity_node_ids:
                    entity_node_ids.add(nid)
                    g.link_chunk_to_node(episode_id, nid)

            relations_added += 1

        stats = {
            "nodes":     len(entity_node_ids),
            "relations": relations_added,
        }
        logger.info(
            "buildKG episode=%s  nodes=%d  relations=%d",
            episode_id, stats["nodes"], stats["relations"],
        )
        return stats


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import shutil
    from db.hrg_graph import HrgGraph, _node_id

    TEST_PATH = "data/hrg_buildkg_test"
    shutil.rmtree(TEST_PATH, ignore_errors=True)

    graph = HrgGraph(TEST_PATH)
    builder = BuildKG(graph)

    entities = ["Prometheus", "Grafana", "Node Exporter", "LLM server"]
    triples = [
        ["Prometheus", "scrapes", "LLM server"],
        ["Prometheus", "includes", "Node Exporter"],
        ["Grafana", "visualises", "Prometheus"],
    ]

    stats = builder.build(entities, triples, episode_id="ep_abc123")
    print(f"build() returned: {stats}")
    print(f"DB nodes     : {graph.node_count}")
    print(f"DB relations : {graph.relation_count}")
    print(f"DB chunks    : {graph.chunk_count}")

    chunks = graph.get_chunks_for_nodes([_node_id("Prometheus"), _node_id("Grafana")])
    print(f"chunks for Prometheus+Grafana: {chunks}")

    graph.clear()
    graph.close()
    shutil.rmtree(TEST_PATH, ignore_errors=True)
    print("Done.")

# ./vevenv/bin/python -m src.HRG.helpers.buildKG
