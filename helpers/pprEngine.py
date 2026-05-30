"""
Steps 3 + 4 + 5 — igraph build, Personalized PageRank, and passage scoring.

Step 3 — build_igraph():
    Loads all HrgRelation + HrgSynonym edges from KuzuDB into an undirected
    igraph Graph. Node IDs are mapped to integers (igraph requirement).

Step 4 — run_ppr():
    Runs igraph's personalized_pagerank() seeded at query nodes.
    Seed weights come from NodeMapping (node specificity normalised).
    Damping factor d=0.5 (from HippoRAG paper).
    Returns {node_id: ppr_score} for all nodes.

Step 5 — score_passages():
    Takes top-K nodes by PPR score → looks up HrgChunkNode links →
    aggregates scores per chunk → returns ranked [(episode_id, score), ...].

run() chains all three steps.

Run:
    ./vevenv/bin/python -m src.HRG.helpers.pprEngine
"""

from __future__ import annotations

import igraph

from logger import get_logger
from db.hrg_graph import HrgGraph

logger = get_logger(__name__)

_DAMPING  = 0.5   # HippoRAG paper value — keeps walk local to seed nodes
_TOP_K_NODES = 20  # top-K nodes used for passage scoring (filters noise)


class PprEngine:

    def __init__(self, graph: HrgGraph) -> None:
        self._graph = graph

    # ── Step 3: Build igraph ─────────────────────────────────────────────────

    def build_igraph(self) -> tuple[igraph.Graph, dict[str, int]]:
        """
        Load KuzuDB edges into an undirected igraph.

        Returns:
            g          — igraph Graph object
            node_index — {node_id: integer_vertex_index}
        """
        logger.info("── STEP 3: Build igraph ──")
        node_index = self._graph.get_node_index()   # {node_id: int}, sorted/stable

        # [PredicateWeight] replace next 2 lines with: edges = self._graph.get_all_edges()
        #                   and update valid/edge_weights lines below accordingly
        edges_with_pred = self._graph.get_all_edges_with_pred()
        pred_weights    = self._graph.get_predicate_weights()

        # Convert string node ID pairs → integer index pairs
        valid       = [(a, b, p) for a, b, p in edges_with_pred if a in node_index and b in node_index]
        int_edges   = [(node_index[a], node_index[b]) for a, b, _ in valid]
        edge_weights = [pred_weights.get(p, 1.0) for _, _, p in valid]  # [PredicateWeight]

        g = igraph.Graph(
            n       = len(node_index),
            edges   = int_edges,
            directed= False,
        )
        g.es['weight'] = edge_weights  # [PredicateWeight] remove if plugin removed

        logger.info(
            "  igraph ready — vertices=%d  edges=%d  (undirected, predicate-weighted)",
            g.vcount(), g.ecount(),
        )
        return g, node_index

    # ── Step 4: Personalized PageRank ────────────────────────────────────────

    def run_ppr(
        self,
        g:          igraph.Graph,
        node_index: dict[str, int],
        seeds:      dict[str, float],
    ) -> dict[str, float]:
        """
        Run personalized PageRank seeded at query entity nodes.

        Args:
            g          — igraph Graph from build_igraph()
            node_index — {node_id: int} mapping from build_igraph()
            seeds      — {node_id: weight} from NodeMapping (weights sum to 1.0)

        Returns:
            {node_id: ppr_score} for every node in the graph.
        """
        logger.info("── STEP 4: Personalized PageRank ── damping=%.1f", _DAMPING)
        n = g.vcount()

        # Build reset vector: seed weights at query entity indices, 0 elsewhere
        reset = [0.0] * n
        for nid, weight in seeds.items():
            if nid in node_index:
                reset[node_index[nid]] = weight

        # Normalise reset vector (should already sum to 1, but guard for safety)
        total = sum(reset)
        if total == 0:
            logger.warning("  PPR: reset vector is all zeros — no valid seed nodes")
            return {}
        reset = [v / total for v in reset]

        # Log reset vector — show only non-zero entries
        logger.info("  Reset vector (seed nodes):")
        for nid, weight in seeds.items():
            if nid in node_index:
                logger.info("    vertex=%-4d  node=%-45s  weight=%.4f",
                            node_index[nid], nid, weight)

        # Run PPR — igraph returns scores in vertex index order
        scores = g.personalized_pagerank(
            damping  = _DAMPING,
            reset    = reset,
            directed = False,
            weights  = 'weight' if g.es.attributes() and 'weight' in g.es.attributes() else None,  # [PredicateWeight]
        )

        # Map integer indices back to node IDs
        index_to_nid = {v: k for k, v in node_index.items()}
        result       = {index_to_nid[i]: scores[i] for i in range(n)}

        top10 = sorted(result.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("  PPR scores (top-10 of %d nodes):", n)
        for rank, (nid, score) in enumerate(top10, 1):
            logger.info("    #%-2d  %-45s  score=%.6f", rank, nid, score)

        return result

    # ── Step 5: Passage scoring ──────────────────────────────────────────────

    def score_passages(
        self,
        node_scores: dict[str, float],
    ) -> list[tuple[str, float]]:
        """
        Translate node PPR scores into passage scores.

        Takes top-K nodes by PPR score → HrgChunkNode lookup →
        sum node scores per chunk → return sorted [(episode_id, score), ...].

        Returns all scored passages sorted descending. Caller applies top_k cutoff.
        """
        logger.info("── STEP 5: Passage Scoring ── top_k_nodes=%d", _TOP_K_NODES)
        if not node_scores:
            return []

        # Top-K nodes only — filters noise from low-scoring nodes
        top_nodes    = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        top_nodes    = top_nodes[:_TOP_K_NODES]
        top_node_ids = [nid for nid, _ in top_nodes]

        # One DB call — get all chunks for all top-K nodes at once
        chunk_map = self._graph.get_chunks_for_nodes(top_node_ids)

        # Aggregate: passage_score += node_ppr_score for every (node, chunk) pair
        passage_scores: dict[str, float] = {}
        logger.info("  Node → episode contributions:")
        for nid, score in top_nodes:
            chunks = chunk_map.get(nid, [])
            if chunks:
                for chunk_id in chunks:
                    passage_scores[chunk_id] = passage_scores.get(chunk_id, 0.0) + score
                logger.info("    %-45s  score=%.6f  → %s", nid, score, chunks)
            else:
                logger.info("    %-45s  score=%.6f  → (no episode link)", nid, score)

        ranked = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)

        logger.info("  Ranked passages (%d total):", len(ranked))
        for ep_id, sc in ranked:
            logger.info("    %-30s  score=%.6f", ep_id, sc)

        return ranked

    # ── Combined entry point ─────────────────────────────────────────────────

    def run(self, seeds: dict[str, float]) -> list[tuple[str, float]]:
        """
        Steps 3 + 4 + 5 in one call.

        Args:
            seeds — {node_id: weight} from NodeMapping

        Returns:
            [(episode_id, score), ...] sorted by score descending.
        """
        if not seeds:
            return []
        g, node_index  = self.build_igraph()
        node_scores    = self.run_ppr(g, node_index, seeds)
        return self.score_passages(node_scores)


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from helpers.nodeMapping import NodeMapping

    graph  = HrgGraph(read_only=True)
    mapper = NodeMapping(graph)
    engine = PprEngine(graph)

    # Simulate a query: "Who delivers cookies to Green Valley School?"
    entities = ["Green Valley School", "Arjun"]
    seeds    = mapper.map(entities)

    print(f"Seeds : {seeds}\n")

    # Steps 3 + 4 individually to show intermediate output
    g, node_index = engine.build_igraph()
    print(f"Step 3 — igraph: {g.vcount()} vertices, {g.ecount()} edges\n")

    node_scores = engine.run_ppr(g, node_index, seeds)
    top10 = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Step 4 — PPR top-10 nodes:")
    for nid, score in top10:
        print(f"  {nid:<45} {score:.6f}")

    # Step 5 — passage scores
    ranked = engine.score_passages(node_scores)
    print(f"\nStep 5 — Ranked passages ({len(ranked)} total):")
    for ep_id, score in ranked:
        print(f"  {ep_id:<30} {score:.6f}")

    mapper._qdrant.close()
    graph.close()
    print("\nDone.")

# ./vevenv/bin/python -m src.HRG.helpers.pprEngine
