"""
HRG Answering — final generation step after retrieval.

Takes ranked passage texts from hrgRetrieval + the original query,
sends them through the RAG QA prompt, and returns a grounded answer.

Usage:
    retrieval = HrgRetrieval()
    answering = HrgAnswering()

    results  = retrieval.retrieve(query, top_k=5)
    passages = [fetch_from_postgres(ep_id) for ep_id, _ in results]
    answer   = answering.answer(query, passages)

Run:
    ./vevenv/bin/python -m src.HRG.hrgAnswering
"""

from __future__ import annotations

from logger import get_logger
from helpers.passageTable import PassageTable
from hrgRetrieval import HrgRetrieval
from llm.client import LLMService
from llm.prompts.rag_qa import build_hrg_qa_prompt, parse_answer

logger = get_logger(__name__)


class HrgAnswering:

    def __init__(self, top_k: int = 5) -> None:
        self._llm       = LLMService()
        self._retrieval = HrgRetrieval()
        self._passages  = PassageTable()
        self._top_k     = top_k

    def answer(self, query: str, passages: list[str]) -> str:
        """
        Generate a grounded answer from retrieved passage texts.

        Args:
            query    — the user's original question
            passages — passage texts fetched from PostgreSQL (ordered by score)

        Returns:
            Concise answer string extracted from the LLM's chain-of-thought response.
        """
        if not passages:
            logger.warning("HrgAnswering: no passages provided")
            return ""

        messages = build_hrg_qa_prompt(query, passages)
        raw      = self._llm.chat(messages, "ragqa")
        logger.info("RAG QA raw response: %s", raw[:300])
        result   = parse_answer(raw)
        logger.info("RAG QA answer: %s", result)
        return result

    def ask(self, query: str) -> str:
        """
        Full HRG pipeline: retrieve → fetch passages → generate answer.

        Args:
            query — natural language question

        Returns:
            Grounded answer string, or empty string if no passages found.
        """
        logger.info("════ HRG ASK  query='%s' ════", query[:80])

        ranked = self._retrieval.retrieve(query, top_k=self._top_k)
        if not ranked:
            logger.warning("HrgAnswering.ask: no episodes retrieved")
            return ""

        fetched  = self._passages.fetch(ranked)
        passages = [text for _, _, text in fetched]

        answer = self.answer(query, passages)
        logger.info("════ HRG ANSWER: %s ════", answer)
        return answer


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    hrg = HrgAnswering(top_k=3)

    queries = [
    # 🟢 Factual / Single-hop
    "When was Elena Vasquez born?",
    "What city did Elena attend boarding school in?",
    # "What subject did Elena major in at MIT?",
    # "Who was Elena's doctoral advisor at Caltech?",
    # "What was the name of Elena's daughter?",
    # "What country was Samuel Okafor originally from?",
    # "What did Rodrigo Vasquez do for work?",
    # "How many siblings did Elena have?",
    # "What journal published Elena's 2018 paper?",
    # "What was the name of Elena's roommate at the New Mexico Academy?",

    # # 🟡 Relational / Two-hop
    # "What connection did Samuel Okafor have to the University of New Mexico?",
    # "What scientific phenomenon did Elena first encounter in Professor Hollis's lab?",
    # "Why did Elena and Daniel's marriage begin to break down?",
    # "What happened to Consuela Vasquez and when?",
    # "Where was the Gruber Prize ceremony held and who did Elena bring?",
    # "What telescope array provided the data that confirmed the Vasquez Hypothesis?",
    # "What did Elena place on her mother's casket at the funeral?",
    # "What institution appointed Elena as director in 2022?",

    # # 🟠 Temporal / Chronological
    # "How old was Sofia when Elena's Nature Astronomy paper was published?",
    # "How many years did Elena spend at MIT before joining Professor Hollis's lab?",
    # "In what year did Elena win the state science competition?",
    # "What happened in Elena's life between 2012 and 2016?",
    # "How long did the SKA precursor monitoring program run before confirming the hypothesis?",

    # # 🔴 Multi-hop / Inference
    # "What trait did Elena share with her daughter Sofia, and how did it manifest differently in each of them?",
    # "How did the advice Elena gave Sofia about problems reflect Elena's own personal history?",
    # "What parallels exist between Elena's childhood in Polvillo and her father's reaction at her wedding?",
    # "How did Elena's response to scientific criticism differ from how most researchers engage in academic disputes?",
    # "In what ways did Consuela silently support Elena's ambitions during her childhood?",

    # # ⚫ Negative / Unanswerable
    # "What was Samuel Okafor's salary at the Polvillo school?",
    # "Did Elena ever collaborate with Priya Mehta on a research project?",
]

    for q in queries:
        print(f"\n{'═'*60}")
        print(f"Q: {q}")
        answer = hrg.ask(q)
        print(f"A: {answer}")

    hrg._retrieval._node_mapper._qdrant.close()
    hrg._retrieval._graph.close()

# ./vevenv/bin/python -m src.HRG.hrgAnswering
