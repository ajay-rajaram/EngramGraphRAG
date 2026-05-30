# RAG QA prompt from HippoRAG paper (Appendix)

_SYSTEM = (
    "As an advanced reading comprehension assistant, your task is to analyze text passages "
    "and corresponding questions meticulously. "
    "Your response starts after 'Thought: ', where you methodically break down the reasoning. "
    "Conclude with 'Answer: ' to present a concise, definitive response."
)

_ONE_SHOT_INPUT = (
    "Wikipedia Title: Southampton\n"
    "The University of Southampton was founded in 1862 and received its Royal Charter in 1952.\n\n"
    "Wikipedia Title: Neville A. Stanton\n"
    "Neville A. Stanton is a British Professor at the University of Southampton.\n\n"
    "Question: When was Neville A. Stanton's employer founded?\nThought: "
)

_ONE_SHOT_OUTPUT = (
    "The employer of Neville A. Stanton is University of Southampton. "
    "The University of Southampton was founded in 1862.\nAnswer: 1862."
)


def build_hrg_qa_prompt(query: str, passages: list[str]) -> list[dict]:
    """Build the RAG QA chat messages from ranked passage texts."""
    context = ""
    for passage in passages:
        context += f"{passage.strip()}\n\n"
    prompt_user = context + f"Question: {query.strip()}\nThought: "

    return [
        {"role": "system",    "content": _SYSTEM},
        {"role": "user",      "content": _ONE_SHOT_INPUT},
        {"role": "assistant", "content": _ONE_SHOT_OUTPUT},
        {"role": "user",      "content": prompt_user},
    ]


def parse_answer(response: str) -> str:
    """Extract text after 'Answer:' marker."""
    if "Answer:" in response:
        return response.split("Answer:", 1)[1].strip()
    return response.strip()
