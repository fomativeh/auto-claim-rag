from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_base_url: str | None
    openai_model: str
    openai_json_mode: bool

    embedding_model: str
    chunk_chars: int
    chunk_overlap_chars: int
    retrieve_top_k: int


def get_settings() -> Settings:
    load_dotenv(override=False)

    return Settings(
        openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
        openai_base_url=os.environ.get("OPENAI_BASE_URL") or None,
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        openai_json_mode=os.environ.get("OPENAI_JSON_MODE", "1") not in {"0", "false", "False"},
        embedding_model=os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        chunk_chars=int(os.environ.get("CHUNK_CHARS", "1400")),
        chunk_overlap_chars=int(os.environ.get("CHUNK_OVERLAP_CHARS", "200")),
        retrieve_top_k=int(os.environ.get("RETRIEVE_TOP_K", "6")),
    )
