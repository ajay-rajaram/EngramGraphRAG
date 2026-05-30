import os
from typing import List, Tuple
from logger import get_logger

logger = get_logger(__name__)

SUPPORTED = {".pdf", ".txt", ".docx"}


class FileProcessor:

    def process(self, input_path: str) -> List[Tuple[str, dict]]:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Path not found: {input_path}")

        if os.path.isfile(input_path):
            file_paths = [input_path]
        else:
            file_paths = [
                os.path.join(root, f)
                for root, _, files in os.walk(input_path)
                for f in files
                if os.path.splitext(f)[1].lower() in SUPPORTED
            ]

        results = []
        for path in file_paths:
            try:
                results.append(self._load(path))
            except Exception as e:
                logger.warning("FileProcessor: skipping '%s' — %s", path, e)
        return results

    def _load(self, file_path: str) -> Tuple[str, dict]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self._load_pdf(file_path)
        elif ext == ".txt":
            return self._load_txt(file_path)
        elif ext in {".docx", ".doc"}:
            return self._load_docx(file_path)
        raise ValueError(f"Unsupported format: {ext}")

    def _load_pdf(self, file_path: str) -> Tuple[str, dict]:
        import fitz
        doc = fitz.open(file_path)
        text = "\n\n".join(p.get_text().strip() for p in doc if p.get_text().strip())
        return text, {"filename": os.path.basename(file_path), "source_type": "pdf"}

    def _load_txt(self, file_path: str) -> Tuple[str, dict]:
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        return text, {"filename": os.path.basename(file_path), "source_type": "txt"}

    def _load_docx(self, file_path: str) -> Tuple[str, dict]:
        from docx import Document
        doc = Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return text, {"filename": os.path.basename(file_path), "source_type": "docx"}
