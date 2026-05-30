_SYSTEM = "You're a very effective entity extraction system."

_EXAMPLE_INPUT = (
    "Please extract all named entities that are important for solving the questions below.\n"
    "Place the named entities in json format.\n\n"
    "Question: Which magazine was started first Arthur's Magazine or First for Women?"
)

_EXAMPLE_OUTPUT = '{"named_entities": ["First for Women", "Arthur\'s Magazine"]}'


def build_query_ner_prompt(query: str) -> list[dict]:
    user_input = (
        "Please extract all named entities that are important for solving the questions below.\n"
        "Place the named entities in json format.\n\n"
        f"Question: {query.strip()}"
    )
    return [
        {"role": "system",    "content": _SYSTEM},
        {"role": "user",      "content": _EXAMPLE_INPUT},
        {"role": "assistant", "content": _EXAMPLE_OUTPUT},
        {"role": "user",      "content": user_input},
    ]
