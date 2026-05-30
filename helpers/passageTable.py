"""
Step 6 — Passage Table.

Owns hrg_test_passages in PostgreSQL — both write (indexing) and read (retrieval).

store() — called by hrgIndexing.process() to persist passage text alongside KG build.
fetch() — called by hrgRetrieval to hydrate ranked episode IDs with text.

Production replacement: swap hrg_test_passages query for episodes table
(source="inquiry" → response field, source="doc_chunk" → text field).

Run:
    ./vevenv/bin/python -m src.HRG.helpers.passageTable
"""

from __future__ import annotations

import psycopg2
from psycopg2.extras import RealDictCursor

from logger import get_logger

logger = get_logger(__name__)

_DB_DSN = "host=localhost port=5432 dbname=memory_aj user=postgres password=postgres"


class PassageTable:

    def store(self, episode_id: str, text: str) -> None:
        """
        Write (episode_id, passage text) to hrg_test_passages.

        Idempotent — ON CONFLICT DO NOTHING so re-indexing the same episode
        doesn't overwrite manually corrected passages.
        """
        with psycopg2.connect(_DB_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO hrg_test_passages (id, passage) VALUES (%s, %s) "
                    "ON CONFLICT (id) DO NOTHING",
                    (episode_id, text),
                )
            conn.commit()
        logger.info("STORED   %-30s  (%d chars)", episode_id, len(text))

    def fetch(
        self,
        ranked: list[tuple[str, float]],
    ) -> list[tuple[str, float, str]]:
        """
        Fetch passage text for ranked episode IDs.

        Args:
            ranked — [(episode_id, score), ...] from PprEngine.score_passages()

        Returns:
            [(episode_id, score, passage_text), ...] — same order, skips missing IDs.
        """
        logger.info("── STEP 6: Fetch Passages ── %d episodes to fetch", len(ranked))
        if not ranked:
            return []

        ep_ids    = [ep_id for ep_id, _ in ranked]
        score_map = {ep_id: score for ep_id, score in ranked}

        with psycopg2.connect(_DB_DSN) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, passage FROM hrg_test_passages WHERE id = ANY(%s)",
                    (ep_ids,),
                )
                rows = {r["id"]: r["passage"] for r in cur.fetchall()}

        result = []
        for ep_id in ep_ids:
            if ep_id in rows:
                text = rows[ep_id]
                result.append((ep_id, score_map[ep_id], text))
                logger.info("  FETCHED  %-30s  (%d chars)", ep_id, len(text))
            else:
                logger.warning("  MISSING  %-30s  not in hrg_test_passages", ep_id)

        logger.info("  Fetched %d / %d passages", len(result), len(ranked))
        return result


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    table = PassageTable()

    test_ranked = [
        ("ep_test002", 0.8553),
        ("ep_test001", 0.8393),
    ]

    passages = table.fetch(test_ranked)

    print(f"\nFetched {len(passages)} passages:\n")
    for ep_id, score, text in passages:
        print(f"[{ep_id}]  score={score:.4f}")
        print(f"  {text}\n")

# ./vevenv/bin/python -m src.HRG.helpers.passageTable
