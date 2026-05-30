import re
from typing import List
from logger import get_logger
from ingest.token_counter import count_tokens

logger = get_logger(__name__)


class ChunkText:

    def chunk(self, text: str, max_tokens: int = 800, overlap_tokens: int = 50) -> List[str]:
        if not text or not text.strip():
            logger.warning("ChunkText: empty text — returning []")
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            s_tokens = count_tokens(sentence)
            if current_tokens + s_tokens > max_tokens and current:
                chunks.append(" ".join(current))
                current = self._overlap(current, overlap_tokens)
                current_tokens = count_tokens(" ".join(current)) if current else 0
            current.append(sentence)
            current_tokens += s_tokens

        if current:
            chunks.append(" ".join(current))

        result = [c.strip() for c in chunks if c.strip()]
        logger.info("ChunkText: %d chunks from %d sentences", len(result), len(sentences))
        return result

    def _overlap(self, sentences: List[str], overlap_tokens: int) -> List[str]:
        if overlap_tokens <= 0 or not sentences:
            return []
        carry: List[str] = []
        used = 0
        for sentence in reversed(sentences):
            t = count_tokens(sentence)
            if used + t <= overlap_tokens:
                carry.insert(0, sentence)
                used += t
            else:
                break
        return carry or [sentences[-1]]

    def _split_sentences(self, text: str) -> List[str]:
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        sentences: List[str] = []
        for para in re.split(r"\n\n+", text):
            para = para.strip()
            if not para:
                continue
            for part in re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", para):
                part = part.strip()
                if part:
                    sentences.append(part)
        return sentences
