import json

from llm.client import LLMService
from llm.prompts.passage_ner import build_ner_prompt
from llm.prompts.query_ner import build_query_ner_prompt
from llm.schemas import SCHEMA_PASSAGE_NER, SCHEMA_QUERY_NER
from logger import get_logger

logger = get_logger(__name__)


class EntityExtraction:

    def __init__(self):
        self._llm = LLMService()

    def passage_entityExtraction(self, text: str) -> list[str]:
        """Extract named entities from a passage (indexing path)."""
        messages = build_ner_prompt(text)
        raw = self._llm.chat(messages, "passagener", format=SCHEMA_PASSAGE_NER)
        logger.info("Passage NER raw: %s", raw[:200])
        return self._parse(raw)

    def query_entityExtraction(self, query: str) -> list[str]:
        """Extract named entities from a query (retrieval path)."""
        messages = build_query_ner_prompt(query)
        raw = self._llm.chat(messages, "queryner", format=SCHEMA_QUERY_NER)
        logger.info("Query NER raw: %s", raw[:200])
        return self._parse(raw)
    
    def _parse(self, raw: str) -> list[str]:
        try:
            return json.loads(raw).get("named_entities", [])
        except (json.JSONDecodeError, AttributeError):
            logger.warning("NER parse failed: %s", raw[:200])
            return []


if __name__ == "__main__":
    entity_extraction = EntityExtraction()
    passage = """
A few days later, Ravi realized that the increased cookie orders from Green Valley School were using flour much faster than expected. To avoid future shortages, he signed a monthly supply agreement with FreshFarm Suppliers for automatic flour deliveries every Tuesday.   
 """
    print(entity_extraction.passage_entityExtraction(passage))


# to run: python -m src.HRG.helpers.entityExtraction