import json

_SYSTEM = (
    "Your task is to construct an RDF (Resource Description Framework) graph "
    "from the given passages and named entity lists.\n"
    "Respond with a JSON list of triples, with each triple representing a relationship.\n"
    "Each triple should contain at least one, preferably two, of the named entities.\n"
    "Clearly resolve pronouns to their specific names.\n"
    "Convert the paragraph into a JSON dict with a named entity list and a triple list."
)

_EXAMPLE_PASSAGE = (
    "Radio City is India's first private FM radio station, started on 3 July 2001. "
    "It plays Hindi, English and regional songs."
)
_EXAMPLE_ENTITIES = ["Radio City", "India", "3 July 2001", "Hindi", "English"]
_EXAMPLE_OUTPUT = json.dumps({
    "triples": [
        ["Radio City", "located in", "India"],
        ["Radio City", "is", "private FM radio station"],
        ["Radio City", "started on", "3 July 2001"],
        ["Radio City", "plays songs in", "Hindi"],
        ["Radio City", "plays songs in", "English"],
    ]
}, indent=2)


def _format_input(passage: str, entities: list[str]) -> str:
    ner_json = json.dumps({"named_entities": entities})
    return (
        f"Convert the paragraph into a JSON dict.\nParagraph:\n```\n{passage.strip()}\n```\n\n"
        f"{ner_json}"
    )


def build_oie_prompt(passage: str, entities: list[str]) -> list[dict]:
    return [
        {"role": "system",    "content": _SYSTEM},
        {"role": "user",      "content": _format_input(_EXAMPLE_PASSAGE, _EXAMPLE_ENTITIES)},
        {"role": "assistant", "content": _EXAMPLE_OUTPUT},
        {"role": "user",      "content": _format_input(passage, entities)},
    ]
