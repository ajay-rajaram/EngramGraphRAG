"""JSON schemas for Ollama structured output."""

SCHEMA_PASSAGE_NER = {
    "type": "object",
    "properties": {
        "named_entities": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
        }
    },
    "required": ["named_entities"],
}

SCHEMA_QUERY_NER = {
    "type": "object",
    "properties": {
        "named_entities": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
        }
    },
    "required": ["named_entities"],
}

SCHEMA_TRIPLES = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 3,
            },
            "minItems": 0,
        }
    },
    "required": ["triples"],
}
