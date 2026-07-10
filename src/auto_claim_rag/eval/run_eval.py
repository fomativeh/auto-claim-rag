from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from auto_claim_rag.embeddings import Embedder
from auto_claim_rag.extraction.extractor import AutoClaimExtractor
from auto_claim_rag.extraction.llm import OpenAIChatLLM, StaticLLM
from auto_claim_rag.extraction.schema import ClaimExtraction
from auto_claim_rag.ingest.loaders import load_documents_from_dir
from auto_claim_rag.retrieval import build_or_load_index
from auto_claim_rag.settings import get_settings
from auto_claim_rag.eval.metrics import aggregate, score_extraction


def _load_gold(path: str) -> dict[str, ClaimExtraction]:
    p = Path(path)
    out: dict[str, ClaimExtraction] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_id = row["doc_id"]
            out[doc_id] = ClaimExtraction.model_validate(row["extraction"])
    return out


def run_eval(
    *,
    claims_dir: str,
    kb_dir: str,
    index_dir: str,
    gold_path: str,
    out_dir: str,
    llm_mode: str,
) -> dict[str, object]:
    settings = get_settings()

    claims = load_documents_from_dir(claims_dir).documents
    gold = _load_gold(gold_path)

    embedder = Embedder(settings.embedding_model)
    kb_index = build_or_load_index(
        kb_dir=kb_dir,
        embedder=embedder,
        chunk_chars=settings.chunk_chars,
        chunk_overlap_chars=settings.chunk_overlap_chars,
        index_dir=index_dir,
    )

    if llm_mode == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for llm_mode=openai")
        llm = OpenAIChatLLM(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            json_mode=settings.openai_json_mode,
        )
    else:
        llm = StaticLLM(
            content=json.dumps(ClaimExtraction().model_dump(), ensure_ascii=False)
        )

    extractor = AutoClaimExtractor(
        llm=llm,
        kb_index=kb_index,
        embedder=embedder,
        retrieve_top_k=settings.retrieve_top_k,
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    preds_path = run_dir / "predictions.jsonl"
    scores_path = run_dir / "scores.jsonl"
    error_analysis_path = run_dir / "error_analysis.json"
    field_confusions_path = run_dir / "field_confusions.json"

    results = []
    mismatch_counts: dict[str, int] = {}
    mismatch_examples: dict[str, list[dict[str, object]]] = {}
    field_pair_counts: dict[str, dict[str, int]] = {}
    with preds_path.open("w", encoding="utf-8") as fp, scores_path.open(
        "w", encoding="utf-8"
    ) as fs:
        for doc in tqdm(claims, desc="extract"):
            pred, retrieved_kb = extractor.extract_with_trace(doc.text)
            fp.write(
                json.dumps(
                    {
                        "doc_id": doc.doc_id,
                        "extraction": pred.model_dump(),
                        "retrieved_kb": retrieved_kb,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            gold_ex = gold.get(doc.doc_id)
            if not gold_ex:
                continue
            scored = score_extraction(doc_id=doc.doc_id, pred=pred, gold=gold_ex)
            results.append(scored)

            for fs_item in scored.field_scores:
                if fs_item.exact_match:
                    continue
                mismatch_counts[fs_item.field] = mismatch_counts.get(fs_item.field, 0) + 1
                ex_list = mismatch_examples.setdefault(fs_item.field, [])
                if len(ex_list) < 5:
                    ex_list.append(
                        {
                            "doc_id": doc.doc_id,
                            "pred_value": getattr(pred, fs_item.field),
                            "gold_value": getattr(gold_ex, fs_item.field),
                        }
                    )

                gold_val = getattr(gold_ex, fs_item.field)
                pred_val = getattr(pred, fs_item.field)
                key = json.dumps(
                    {"gold_value": gold_val, "pred_value": pred_val},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                by_field = field_pair_counts.setdefault(fs_item.field, {})
                by_field[key] = by_field.get(key, 0) + 1

            fs.write(
                json.dumps(
                    {
                        "doc_id": doc.doc_id,
                        "doc_exact_match": scored.exact_match,
                        "field_scores": [
                            {"field": x.field, "exact_match": x.exact_match}
                            for x in scored.field_scores
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = aggregate(results)
    summary["llm_mode"] = llm_mode
    summary["openai_json_mode"] = settings.openai_json_mode
    summary["run_dir"] = str(run_dir)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    top_fields = sorted(
        mismatch_counts.items(), key=lambda kv: (-kv[1], kv[0])
    )
    top_field_names = [k for k, _ in top_fields[:5]]
    error_analysis = {
        "mismatch_counts": mismatch_counts,
        "top_mismatched_fields": top_field_names,
        "examples": {k: mismatch_examples.get(k, []) for k in top_field_names},
    }
    error_analysis_path.write_text(
        json.dumps(error_analysis, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    field_confusions: dict[str, list[dict[str, object]]] = {}
    for field, pair_counts in field_pair_counts.items():
        items = sorted(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        out_items: list[dict[str, object]] = []
        for key, count in items:
            pair = json.loads(key)
            out_items.append(
                {
                    "gold_value": pair["gold_value"],
                    "pred_value": pair["pred_value"],
                    "count": count,
                }
            )
        field_confusions[field] = out_items

    field_confusions_path.write_text(
        json.dumps(field_confusions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims-dir", default="data/raw/claims")
    parser.add_argument("--kb-dir", default="data/knowledge_base")
    parser.add_argument("--index-dir", default="data/index/kb")
    parser.add_argument("--gold-path", default="data/gold/claims_gold.jsonl")
    parser.add_argument("--out-dir", default="data/runs")
    parser.add_argument("--llm", default="openai", choices=["openai", "null"])
    args = parser.parse_args()

    summary = run_eval(
        claims_dir=args.claims_dir,
        kb_dir=args.kb_dir,
        index_dir=args.index_dir,
        gold_path=args.gold_path,
        out_dir=args.out_dir,
        llm_mode=args.llm,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
