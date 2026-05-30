import json

from llm.client import LLMService
from llm.prompts.oie import build_oie_prompt
from llm.schemas import SCHEMA_TRIPLES
from logger import get_logger

logger = get_logger(__name__)


class TripletExtraction:

    def __init__(self):
        self._llm = LLMService()

    def tripletExtraction(self, text: str, entities: list[str]) -> list[list[str]]:
        """Extract (subject, predicate, object) triples from passage + NER entity list."""
        messages = build_oie_prompt(text, entities)
        raw = self._llm.chat(messages, "passagener", format=SCHEMA_TRIPLES)
        logger.info("OIE raw output: %s", raw[:300])
        return self._parse(raw)

    def _parse(self, raw: str) -> list[list[str]]:
        try:
            triples = json.loads(raw).get("triples", [])
            return self._filter(triples)
        except (json.JSONDecodeError, AttributeError):
            logger.warning("OIE parse failed: %s", raw[:300])
            return []

    def _filter(self, triples: list) -> list[list[str]]:
        """Keep only well-formed unique [subject, predicate, object] triples."""
        seen = set()
        valid = []
        for t in triples:
            if not isinstance(t, (list, tuple)) or len(t) != 3:
                continue
            triple = [str(x).strip() for x in t]
            key = tuple(triple)
            if key not in seen and all(key):
                seen.add(key)
                valid.append(triple)
        return valid


if __name__ == "__main__":
    triplet_extraction = TripletExtraction()
    passage = """
    A few days later, Ravi realized that the increased cookie orders from Green Valley School were using flour much faster than expected. To avoid future shortages, he signed a monthly supply agreement with FreshFarm Suppliers for automatic flour deliveries every Tuesday.   
    """
    entities = ["Ravi", "Green Valley School", "FreshFarm Suppliers"]
    print(triplet_extraction.tripletExtraction(passage, entities))


# to run: python -m src.HRG.helpers.tripletExtraction