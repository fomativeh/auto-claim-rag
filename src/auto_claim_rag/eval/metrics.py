from __future__ import annotations

from dataclasses import dataclass
import re

from auto_claim_rag.extraction.schema import ClaimExtraction


def _norm_str(x: str | None) -> str | None:
    if x is None:
        return None
    s = " ".join(x.strip().split())
    return s.lower() if s else None


def _digits_only(x: str) -> str:
    return "".join(ch for ch in x if ch.isdigit())


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def _tokenize(x: str) -> set[str]:
    x = _norm_str(x) or ""
    x = re.sub(r"[^a-z0-9]+", " ", x)
    tokens = [t for t in x.split() if t and t not in _STOPWORDS]
    return set(tokens)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _eq_field(field: str, a: object, b: object) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    if field == "loss_description" and isinstance(a, str) and isinstance(b, str):
        ta = _tokenize(a)
        tb = _tokenize(b)
        return _jaccard(ta, tb) >= 0.5

    if field == "contact_phone" and isinstance(a, str) and isinstance(b, str):
        return _digits_only(a) == _digits_only(b)

    if field == "contact_email" and isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()

    if isinstance(a, str) and isinstance(b, str):
        return _norm_str(a) == _norm_str(b)
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1.0
    return a == b


@dataclass(frozen=True)
class FieldScore:
    field: str
    exact_match: bool


@dataclass(frozen=True)
class EvalResult:
    doc_id: str
    field_scores: list[FieldScore]

    @property
    def exact_match(self) -> bool:
        return all(fs.exact_match for fs in self.field_scores)


def score_extraction(*, doc_id: str, pred: ClaimExtraction, gold: ClaimExtraction) -> EvalResult:
    fields = list(ClaimExtraction.model_fields.keys())
    out: list[FieldScore] = []
    for f in fields:
        out.append(
            FieldScore(
                field=f,
                exact_match=_eq_field(f, getattr(pred, f), getattr(gold, f)),
            )
        )
    return EvalResult(doc_id=doc_id, field_scores=out)


def aggregate(results: list[EvalResult]) -> dict[str, float]:
    if not results:
        return {"doc_exact_match": 0.0, "field_accuracy": 0.0}

    doc_em = sum(1 for r in results if r.exact_match) / len(results)

    total_fields = sum(len(r.field_scores) for r in results)
    correct_fields = sum(1 for r in results for fs in r.field_scores if fs.exact_match)
    field_acc = correct_fields / total_fields if total_fields else 0.0

    return {"doc_exact_match": doc_em, "field_accuracy": field_acc}
