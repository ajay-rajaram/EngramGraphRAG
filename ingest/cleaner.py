import re
from collections import Counter
from logger import get_logger

logger = get_logger(__name__)

_REFERENCES_RE = re.compile(
    r"(?m)^\s*(?:References|Bibliography|Works Cited|Reference List)\s*$",
    re.IGNORECASE,
)


class CleanText:

    def clean(self, text: str) -> str:
        original_len = len(text)
        text = self._normalize_whitespace(text)
        text = self._strip_references_section(text)
        text = self._strip_page_numbers(text)
        text = self._strip_repeated_headers_footers(text)
        text = self._collapse_blank_lines(text)
        result = text.strip()
        logger.info("CleanText: %d chars in -> %d chars out", original_len, len(result))
        return result

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\r", "\n", text)
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r" +\n", "\n", text)
        return text

    def _strip_references_section(self, text: str) -> str:
        m = _REFERENCES_RE.search(text)
        return text[:m.start()].rstrip() if m else text

    def _strip_page_numbers(self, text: str) -> str:
        return re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)

    def _strip_repeated_headers_footers(self, text: str) -> str:
        lines = text.split("\n")
        counts = Counter(line.strip() for line in lines if line.strip())
        repeated = {line for line, count in counts.items() if count >= 3}
        return "\n".join(line for line in lines if line.strip() not in repeated)

    def _collapse_blank_lines(self, text: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", text)
