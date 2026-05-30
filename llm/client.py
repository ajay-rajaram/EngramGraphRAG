"""
LLMService — thin Ollama wrapper.
Configure model aliases in config.toml under [ollama].
"""

import toml
import ollama
from logger import get_logger

config = toml.load("config.toml")
logger = get_logger("LLMService")


class LLMService:

    def chat(self, messages: list[dict], model_alias: str = "ragqa") -> str:
        """
        Send a multi-turn chat to Ollama.

        Args:
            messages     — list of {role, content} dicts
            model_alias  — key in config.toml [ollama] section
        """
        model = config["ollama"].get(model_alias, config["ollama"]["default"])
        logger.info("chat() model=%s turns=%d", model, len(messages))
        response = ollama.chat(model=model, messages=messages)
        return response["message"]["content"]

    def chat_structured(self, messages: list[dict], model_alias: str, schema: dict) -> str:
        """Same as chat() but enforces JSON output via Ollama's format parameter."""
        model = config["ollama"].get(model_alias, config["ollama"]["default"])
        logger.info("chat() model=%s turns=%d structured=True", model, len(messages))
        response = ollama.chat(model=model, messages=messages, format=schema)
        return response["message"]["content"]
