from __future__ import annotations

import argparse
import json
from pathlib import Path

from pypdf import PdfReader

from auto_claim_rag.embeddings import Embedder
from auto_claim_rag.extraction.extractor import AutoClaimExtractor
from auto_claim_rag.extraction.llm import OpenAIChatLLM
from auto_claim_rag.ingest.loaders import load_documents_from_dir
from auto_claim_rag.retrieval import build_index, build_or_load_index
from auto_claim_rag.settings import get_settings


def _load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _load_single_doc_text(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        reader = PdfReader(str(p))
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages).strip()
    return _load_text(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index")
    p_index.add_argument("--kb-dir", default="data/knowledge_base")
    p_index.add_argument("--index-dir", default="data/index/kb")

    p_latest = sub.add_parser("latest-run")

    p_extract = sub.add_parser("extract")
    p_extract.add_argument("--doc", required=True, help="Path to a .txt or .pdf file")
    p_extract.add_argument("--kb-dir", default="data/knowledge_base")
    p_extract.add_argument("--index-dir", default="data/index/kb")

    p_batch = sub.add_parser("batch-extract")
    p_batch.add_argument("--claims-dir", default="data/raw/claims")
    p_batch.add_argument("--kb-dir", default="data/knowledge_base")
    p_batch.add_argument("--index-dir", default="data/index/kb")
    p_batch.add_argument("--out", default="data/outputs/predictions.jsonl")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--claims-dir", default="data/raw/claims")
    p_eval.add_argument("--kb-dir", default="data/knowledge_base")
    p_eval.add_argument("--index-dir", default="data/index/kb")
    p_eval.add_argument("--gold-path", default="data/gold/claims_gold.jsonl")
    p_eval.add_argument("--out-dir", default="data/runs")
    p_eval.add_argument("--llm", default="openai", choices=["openai", "null"])

    args = parser.parse_args()

    settings = get_settings()

    if args.cmd == "index":
        kb_docs = load_documents_from_dir(args.kb_dir).documents
        embedder = Embedder(settings.embedding_model)
        kb_index = build_index(
            kb_docs,
            embedder=embedder,
            chunk_chars=settings.chunk_chars,
            chunk_overlap_chars=settings.chunk_overlap_chars,
        )
        kb_index.save(args.index_dir)
        print(json.dumps({"index_dir": args.index_dir, "chunks": len(kb_index.chunks)}, indent=2))
        return

    if args.cmd == "latest-run":
        runs_dir = Path("data/runs")
        if not runs_dir.exists():
            raise FileNotFoundError(str(runs_dir))

        candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
        if not candidates:
            raise FileNotFoundError("No runs found in data/runs")

        latest = sorted(candidates, key=lambda p: p.name)[-1]
        summary_path = latest / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(str(summary_path))

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {"summary_path": str(summary_path), "summary": summary},
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if args.cmd == "eval":
        from auto_claim_rag.eval.run_eval import run_eval

        summary = run_eval(
            claims_dir=args.claims_dir,
            kb_dir=args.kb_dir,
            index_dir=args.index_dir,
            gold_path=args.gold_path,
            out_dir=args.out_dir,
            llm_mode=args.llm,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.cmd == "batch-extract":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for batch extraction")

        embedder = Embedder(settings.embedding_model)
        kb_index = build_or_load_index(
            kb_dir=args.kb_dir,
            embedder=embedder,
            chunk_chars=settings.chunk_chars,
            chunk_overlap_chars=settings.chunk_overlap_chars,
            index_dir=args.index_dir,
        )

        llm = OpenAIChatLLM(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            json_mode=settings.openai_json_mode,
        )

        extractor = AutoClaimExtractor(
            llm=llm,
            kb_index=kb_index,
            embedder=embedder,
            retrieve_top_k=settings.retrieve_top_k,
        )

        docs = load_documents_from_dir(args.claims_dir).documents
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for doc in docs:
                pred, retrieved_kb = extractor.extract_with_trace(doc.text)
                f.write(
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

        print(json.dumps({"out": str(out_path), "docs": len(docs)}, indent=2))
        return

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for extraction")

    embedder = Embedder(settings.embedding_model)
    kb_index = build_or_load_index(
        kb_dir=args.kb_dir,
        embedder=embedder,
        chunk_chars=settings.chunk_chars,
        chunk_overlap_chars=settings.chunk_overlap_chars,
        index_dir=args.index_dir,
    )

    llm = OpenAIChatLLM(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        base_url=settings.openai_base_url,
        json_mode=settings.openai_json_mode,
    )

    extractor = AutoClaimExtractor(
        llm=llm,
        kb_index=kb_index,
        embedder=embedder,
        retrieve_top_k=settings.retrieve_top_k,
    )
    text = _load_single_doc_text(args.doc)
    out = extractor.extract(document_text=text)
    print(json.dumps(out.model_dump(), indent=2))


if __name__ == "__main__":
    main()
