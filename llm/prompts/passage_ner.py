_SYSTEM = (
    "Your task is to extract named entities from the given paragraph. "
    "Respond with a JSON list of entities."
)

_EXAMPLE_PASSAGE = (
    "Radio City is India's first private FM radio station and was started on 3 July 2001. "
    "It plays Hindi, English and regional songs."
)

_EXAMPLE_OUTPUT = (
    '{"named_entities": ["Radio City", "India", "3 July 2001", "Hindi", "English"]}'
)


def build_ner_prompt(passage: str) -> list[dict]:
    return [
        {"role": "system",    "content": _SYSTEM},
        {"role": "user",      "content": _EXAMPLE_PASSAGE},
        {"role": "assistant", "content": _EXAMPLE_OUTPUT},
        {"role": "user",      "content": passage.strip()},
    ]
