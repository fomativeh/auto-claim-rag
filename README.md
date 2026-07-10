# Auto Insurance Claim Extraction (RAG + Structured Output)

An end-to-end tool that ingests messy auto-insurance claim documents (intake notes, estimates, emails) and extracts structured fields via prompting, with retrieval over a small domain knowledge base (glossary + policy snippets).

Vertical: auto insurance  
Task: claim intake extraction into a fixed schema

## Why a RAG layer for extraction

Extraction accuracy often fails when the model misreads domain shorthand (FNOL, DOL, PD/BI), mixes up insured vs claimant roles, or guesses coverage type. A small, curated knowledge base gives consistent definitions and mapping rules, and retrieval makes those available at extraction time.

## Scope

This repository focuses on a small extraction pipeline for auto-claim artifacts. It parses plain-text claim documents (and can read PDFs when they contain extractable text), retrieves supporting context from a tiny knowledge base, and produces a single schema-validated JSON extraction per document.

Included:

- Schema-first extraction with Pydantic validation
- Paragraph-aware chunking and dense retrieval over glossary/policy excerpts
- A small gold set plus an evaluation script that writes predictions and metrics to a timestamped run folder

Not included:

- OCR for scanned PDFs
- Line-item/table reconstruction from estimates
- Hybrid retrieval/reranking and retrieval-ground-truth evaluation (e.g., recall@k on labeled evidence)
- Review UI/workflow, PII redaction, and other production hardening
- Background jobs and streaming ingestion

## Directory layout

```
data/
  raw/claims/                # messy sample claim docs (.txt)
  knowledge_base/            # glossary + policy snippets used for retrieval
  gold/claims_gold.jsonl      # gold extractions (doc_id -> structured fields)
  runs/                       # eval outputs (predictions + scores)
src/auto_claim_rag/           # ingestion -> chunking -> retrieval -> extraction -> eval
```

## Quickstart

Python 3.10+

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Set env vars:

```bash
copy .env.example .env
```

Then populate `OPENAI_API_KEY` (or point `OPENAI_BASE_URL` at an OpenAI-compatible endpoint).

## Run extraction on one document

```bash
python -m auto_claim_rag extract --doc data/raw/claims/claim_001_fnol.txt
```

## Batch extract a folder

```bash
python -m auto_claim_rag batch-extract --claims-dir data/raw/claims --out data/outputs/predictions.jsonl
```

## Run evaluation (gold set)

```bash
python -m auto_claim_rag eval --llm openai
```

If you want to smoke-test the pipeline without an LLM API key, you can run:

```bash
python -m auto_claim_rag eval --llm null
```

Outputs are written to `data/runs/<timestamp>/`.
