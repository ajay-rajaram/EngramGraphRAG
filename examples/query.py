"""
Example: Query EngramGraphRAG with a natural language question.

Run from the EngramGraphRAG directory:
    python examples/query.py "What is the capital of France?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hrgAnswering import HrgAnswering


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/query.py \"your question here\"")
        sys.exit(1)

    query = sys.argv[1]
    hrg   = HrgAnswering(top_k=3)

    print(f"\nQuery : {query}")
    answer = hrg.ask(query)
    print(f"Answer: {answer}\n")

    hrg._retrieval._node_mapper._qdrant.close()
    hrg._retrieval._graph.close()
