from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from auto_claim_rag.types import Document


@dataclass(frozen=True)
class LoadResult:
    documents: list[Document]
    skipped_paths: list[str]


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)
    return "\n\n".join(pages)


def load_documents_from_dir(dir_path: str) -> LoadResult:
    root = Path(dir_path)
    if not root.exists():
        raise FileNotFoundError(dir_path)

    documents: list[Document] = []
    skipped: list[str] = []

    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if path.suffix.lower() not in {".txt", ".pdf"}:
            skipped.append(str(path))
            continue

        if path.suffix.lower() == ".txt":
            text = _read_text_file(path)
        else:
            text = _read_pdf_file(path)

        doc_id = path.stem
        documents.append(Document(doc_id=doc_id, source_path=str(path), text=text))

    return LoadResult(documents=documents, skipped_paths=skipped)
