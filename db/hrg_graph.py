"""
HRG Graph Database — isolated KuzuDB store for HippoRAG indexing.

Completely separate from VortexEO's existing data/graph/ database.
Only this file talks to KuzuDB for HRG. All other HRG modules call
methods here — they never touch KuzuDB directly.

Schema (5 tables in data/hrg/):
    HrgNode         — noun phrases / named entities (nodes)
    HrgChunk        — passages / episodes (nodes)
    HrgRelation     — OpenIE triples  (HrgNode → HrgNode)
    HrgSynonym      — similarity bridges (HrgNode → HrgNode)
    HrgChunkNode    — passage-node matrix (HrgChunk → HrgNode)

Run:
    ./vevenv/bin/python -m src.HRG.db.hrg_graph
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Optional

import kuzu

from logger import get_logger

logger = get_logger(__name__)

_DEFAULT_PATH = "data/hrg_store/database.kz"


def _normalize(text: str) -> str:
    """Canonical form used for node identity: lowercase, collapse whitespace+underscores to one space."""
    return re.sub(r"[\s_]+", " ", text.lower().strip())


def _node_id(text: str) -> str:
    """Deterministic node ID derived from the normalised text.

    'centralized logging' and 'centralized_logging' produce the same ID.
    """
    norm = _normalize(text)
    slug = norm[:40].replace(" ", "_").replace("/", "_").replace(".", "")
    h    = hashlib.md5(norm.encode()).hexdigest()[:6]
    return f"{slug}_{h}"


class HrgGraph:
    """
    Isolated KuzuDB graph for HRG.

    Write path (indexing):
        upsert_node, add_relation, add_chunk,
        link_chunk_to_node, add_synonym

    Read path (retrieval / PPR):
        get_all_edges, get_node_index,
        get_chunks_for_nodes, get_node_by_text
    """

    def __init__(self, db_path: str = _DEFAULT_PATH, read_only: bool = False) -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._db   = kuzu.Database(db_path, read_only=read_only)
        self._conn = kuzu.Connection(self._db)
        if not read_only:
            self._ensure_schema()
        logger.info("HrgGraph ready at '%s' [%s] — %d nodes  %d relation edges",
                    db_path, "ro" if read_only else "rw", self.node_count, self.relation_count)

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        """Create all 5 tables if they don't exist."""
        # Node tables first — rel tables reference them
        self._try(
            "CREATE NODE TABLE HrgNode("
            "id    STRING, "
            "text  STRING, "
            "PRIMARY KEY(id))"
        )
        self._try(
            "CREATE NODE TABLE HrgChunk("
            "id  STRING, "
            "PRIMARY KEY(id))"
        )
        # Edge tables
        self._try(
            "CREATE REL TABLE HrgRelation("
            "FROM HrgNode TO HrgNode, "
            "predicate    STRING, "
            "chunk_id     STRING)"
        )
        self._try(
            "CREATE REL TABLE HrgSynonym("
            "FROM HrgNode TO HrgNode, "
            "cosine_score DOUBLE)"
        )
        self._try(
            "CREATE REL TABLE HrgChunkNode("
            "FROM HrgChunk TO HrgNode)"
        )

    def _try(self, ddl: str) -> None:
        try:
            self._conn.execute(ddl)
        except RuntimeError as e:
            if "already exists" not in str(e).lower():
                raise

    # ── Write — nodes ──────────────────────────────────────────────────────────

    def upsert_node(self, text: str) -> str:
        """
        Create HrgNode if it doesn't exist.

        Returns the node id. Safe to call multiple times for the same text.
        """
        nid = _node_id(text)
        exists = self._conn.execute(
            "MATCH (n:HrgNode {id: $id}) RETURN n.id",
            parameters={"id": nid},
        )
        if not exists.has_next():
            self._conn.execute(
                "CREATE (n:HrgNode {id: $id, text: $text})",
                parameters={"id": nid, "text": text.strip()},
            )
            logger.debug("HrgNode created: '%s' → %s", text.strip(), nid)
        else:
            norm = _normalize(text)
            raw  = text.strip().lower()
            if norm != raw:
                logger.debug(
                    "HrgNode dedup [normalized]: '%s' → '%s' merged into %s",
                    text.strip(), norm, nid,
                )
            else:
                logger.debug("HrgNode dedup [exact]: '%s' → %s", text.strip(), nid)
        return nid

    def chunk_exists(self, chunk_id: str) -> bool:
        """Return True if this episode_id has already been indexed."""
        r = self._conn.execute(
            "MATCH (c:HrgChunk {id: $id}) RETURN c.id",
            parameters={"id": chunk_id},
        )
        return r.has_next()

    def add_chunk(self, chunk_id: str) -> None:
        """Create HrgChunk node for a passage/episode. Idempotent."""
        exists = self._conn.execute(
            "MATCH (c:HrgChunk {id: $id}) RETURN c.id",
            parameters={"id": chunk_id},
        )
        if not exists.has_next():
            self._conn.execute(
                "CREATE (c:HrgChunk {id: $id})",
                parameters={"id": chunk_id},
            )
            logger.debug("HrgChunk created: %s", chunk_id)

    # ── Write — edges ──────────────────────────────────────────────────────────

    def add_relation(
        self,
        subj: str,
        pred: str,
        obj:  str,
        chunk_id: str,
    ) -> None:
        """
        Add HrgRelation edge between subject and object nodes.

        Upserts both nodes first. Skips duplicate (subj, pred, obj) triples.
        """
        sid = self.upsert_node(subj)
        oid = self.upsert_node(obj)

        exists = self._conn.execute(
            "MATCH (a:HrgNode {id: $sid})-[r:HrgRelation {predicate: $pred}]->(b:HrgNode {id: $oid}) "
            "RETURN r.predicate",
            parameters={"sid": sid, "pred": pred, "oid": oid},
        )
        if not exists.has_next():
            self._conn.execute(
                "MATCH (a:HrgNode {id: $sid}), (b:HrgNode {id: $oid}) "
                "CREATE (a)-[:HrgRelation {predicate: $pred, chunk_id: $cid}]->(b)",
                parameters={"sid": sid, "oid": oid, "pred": pred, "cid": chunk_id},
            )
            logger.debug("HrgRelation: '%s' -[%s]-> '%s'", subj, pred, obj)

    def link_chunk_to_node(self, chunk_id: str, node_id: str) -> None:
        """
        Add HrgChunkNode edge — records that chunk contains this node.

        This is the passage-node matrix entry. Idempotent.
        """
        exists = self._conn.execute(
            "MATCH (c:HrgChunk {id: $cid})-[:HrgChunkNode]->(n:HrgNode {id: $nid}) "
            "RETURN c.id",
            parameters={"cid": chunk_id, "nid": node_id},
        )
        if not exists.has_next():
            self._conn.execute(
                "MATCH (c:HrgChunk {id: $cid}), (n:HrgNode {id: $nid}) "
                "CREATE (c)-[:HrgChunkNode]->(n)",
                parameters={"cid": chunk_id, "nid": node_id},
            )

    def add_synonym(
        self,
        node_id_a:    str,
        node_id_b:    str,
        cosine_score: float,
    ) -> None:
        """
        Add HrgSynonym edge between two similar nodes.

        Called by synonymEdges.py after Qdrant similarity search.
        Skips if edge already exists in either direction.
        """
        exists = self._conn.execute(
            "MATCH (a:HrgNode {id: $aid})-[:HrgSynonym]-(b:HrgNode {id: $bid}) "
            "RETURN a.id",
            parameters={"aid": node_id_a, "bid": node_id_b},
        )
        if not exists.has_next():
            self._conn.execute(
                "MATCH (a:HrgNode {id: $aid}), (b:HrgNode {id: $bid}) "
                "CREATE (a)-[:HrgSynonym {cosine_score: $score}]->(b)",
                parameters={"aid": node_id_a, "bid": node_id_b, "score": cosine_score},
            )
            logger.debug("HrgSynonym: %s ↔ %s (%.3f)", node_id_a, node_id_b, cosine_score)

    # ── Read — retrieval / PPR ─────────────────────────────────────────────────

    def get_all_edges(self) -> list[tuple[str, str]]:
        """
        Return all edges (HrgRelation + HrgSynonym) as (node_id_a, node_id_b) pairs.

        Used to build the igraph object for PPR computation.
        """
        edges = []
        # get all relation edges
        result = self._conn.execute(
            "MATCH (a:HrgNode)-[:HrgRelation]->(b:HrgNode) RETURN a.id, b.id"
        )
        # get all relation edges
        while result.has_next():
            row = result.get_next()
            edges.append((row[0], row[1]))

        result = self._conn.execute(
            "MATCH (a:HrgNode)-[:HrgSynonym]->(b:HrgNode) RETURN a.id, b.id"
        )
        while result.has_next():
            row = result.get_next()
            edges.append((row[0], row[1]))

        return edges

    # [PredicateWeight] remove both methods below if plugin removed
    def get_all_edges_with_pred(self) -> list[tuple[str, str, str]]:
        """Return (node_id_a, node_id_b, predicate) for HrgRelation edges.
        HrgSynonym edges are returned with predicate='__synonym__'."""
        edges = []
        result = self._conn.execute(
            "MATCH (a:HrgNode)-[r:HrgRelation]->(b:HrgNode) "
            "RETURN a.id, b.id, r.predicate"
        )
        while result.has_next():
            row = result.get_next()
            edges.append((row[0], row[1], row[2]))

        result = self._conn.execute(
            "MATCH (a:HrgNode)-[:HrgSynonym]->(b:HrgNode) RETURN a.id, b.id"
        )
        while result.has_next():
            row = result.get_next()
            edges.append((row[0], row[1], "__synonym__"))

        return edges

    def get_predicate_weights(self) -> dict[str, float]:
        """Return {predicate: 1/frequency} for all HrgRelation predicates.
        Synonym edges are assigned weight 1.0 (treated as specific)."""
        result = self._conn.execute(
            "MATCH ()-[r:HrgRelation]->() "
            "RETURN r.predicate, count(r) AS freq"
        )
        weights: dict[str, float] = {}
        while result.has_next():
            row = result.get_next()
            predicate, freq = row[0], row[1]
            weights[predicate] = 1.0 / max(freq, 1)
        weights["__synonym__"] = 1.0
        return weights

    def get_node_index(self) -> dict[str, int]:
        """
        Return {node_id: integer_index} for all HrgNodes.

        igraph needs integer vertex indices — this maps string IDs to ints.
        The mapping is stable within a session (sorted for determinism).
        """
        result = self._conn.execute(
            "MATCH (n:HrgNode) RETURN n.id ORDER BY n.id"
        )
        index = {}
        i = 0
        while result.has_next():
            index[result.get_next()[0]] = i
            i += 1
        return index

    def get_chunks_for_nodes(
        self,
        node_ids: list[str],
    ) -> dict[str, list[str]]:
        """
        Return {node_id: [chunk_id, ...]} for the given node IDs.

        Used after PPR: high-scoring nodes → which chunks contain them?
        """
        if not node_ids:
            return {}

        mapping: dict[str, list[str]] = {nid: [] for nid in node_ids}
        for nid in node_ids:
            result = self._conn.execute(
                "MATCH (c:HrgChunk)-[:HrgChunkNode]->(n:HrgNode {id: $nid}) "
                "RETURN c.id",
                parameters={"nid": nid},
            )
            while result.has_next():
                mapping[nid].append(result.get_next()[0])
        return mapping

    def get_all_nodes(self) -> dict[str, str]:
        """Return {node_id: text} for all HrgNodes. Used by synonymEdges to embed."""
        result = self._conn.execute(
            "MATCH (n:HrgNode) RETURN n.id, n.text ORDER BY n.id"
        )
        nodes: dict[str, str] = {}
        while result.has_next():
            row = result.get_next()
            nodes[row[0]] = row[1]
        return nodes

    def get_node_by_text(self, text: str) -> Optional[str]:
        """
        Exact text lookup → node_id.

        Used at retrieval time to map a query entity to its graph node.
        Returns None if not found.
        """
        nid    = _node_id(text)
        result = self._conn.execute(
            "MATCH (n:HrgNode {id: $id}) RETURN n.id",
            parameters={"id": nid},
        )
        return result.get_next()[0] if result.has_next() else None

    # ── Stats & maintenance ────────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        r = self._conn.execute("MATCH (n:HrgNode) RETURN count(n)")
        return r.get_next()[0] if r.has_next() else 0

    @property
    def relation_count(self) -> int:
        r = self._conn.execute("MATCH ()-[r:HrgRelation]->() RETURN count(r)")
        return r.get_next()[0] if r.has_next() else 0

    @property
    def synonym_count(self) -> int:
        r = self._conn.execute("MATCH ()-[r:HrgSynonym]->() RETURN count(r)")
        return r.get_next()[0] if r.has_next() else 0

    @property
    def chunk_count(self) -> int:
        r = self._conn.execute("MATCH (c:HrgChunk) RETURN count(c)")
        return r.get_next()[0] if r.has_next() else 0

    def clear(self) -> None:
        """Delete all data. Edges must be deleted before nodes in KuzuDB."""
        self._conn.execute("MATCH ()-[r:HrgChunkNode]->() DELETE r")
        self._conn.execute("MATCH ()-[r:HrgSynonym]->() DELETE r")
        self._conn.execute("MATCH ()-[r:HrgRelation]->() DELETE r")
        self._conn.execute("MATCH (c:HrgChunk) DELETE c")
        self._conn.execute("MATCH (n:HrgNode) DELETE n")
        logger.info("HrgGraph cleared")

    def close(self) -> None:
        self._conn.close()


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import shutil
    TEST_PATH = "data/hrg_store/database.kz"
    shutil.rmtree("data/hrg_store", ignore_errors=True)

    g = HrgGraph(TEST_PATH)

    # Write nodes + relations
    g.add_chunk("chunk_001")
    g.add_relation("Prometheus", "scrapes", "LLM server", "chunk_001")
    g.add_relation("Prometheus", "includes", "Node Exporter", "chunk_001")
    g.add_relation("Grafana", "visualises", "Prometheus", "chunk_001")

    # Link chunk → nodes
    for text in ["Prometheus", "LLM server", "Node Exporter", "Grafana"]:
        nid = g.upsert_node(text)
        g.link_chunk_to_node("chunk_001", nid)

    # Synonym edge
    nid_a = g._node_id("Node Exporter") if hasattr(g, "_node_id") else _node_id("Node Exporter")
    nid_b = g._node_id("Blackbox Exporter") if hasattr(g, "_node_id") else _node_id("Blackbox Exporter")
    g.upsert_node("Blackbox Exporter")
    g.add_synonym(_node_id("Node Exporter"), _node_id("Blackbox Exporter"), 0.87)

    print(f"nodes     : {g.node_count}")
    print(f"relations : {g.relation_count}")
    print(f"synonyms  : {g.synonym_count}")
    print(f"chunks    : {g.chunk_count}")

    # Read edges for igraph
    edges = g.get_all_edges()
    print(f"\nall edges ({len(edges)}):")
    for e in edges:
        print(f"  {e[0]}  →  {e[1]}")

    # Chunk lookup
    top_nodes = [_node_id("Prometheus"), _node_id("Grafana")]
    mapping   = g.get_chunks_for_nodes(top_nodes)
    print(f"\nchunks for top nodes:")
    for nid, chunks in mapping.items():
        print(f"  {nid} → {chunks}")

    # Node index
    idx = g.get_node_index()
    print(f"\nnode index: {idx}")

    g.clear()
    g.close()
    shutil.rmtree("data/hrg_store", ignore_errors=True)
    print("\nDone.")

# ./vevenv/bin/python -m src.HRG.db.hrg_graph
