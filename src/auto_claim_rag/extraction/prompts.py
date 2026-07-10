from __future__ import annotations

import json

from auto_claim_rag.extraction.schema import ClaimExtraction


def claim_json_schema() -> str:
    schema = ClaimExtraction.model_json_schema()
    return json.dumps(schema, ensure_ascii=False)


def build_claim_extraction_prompt(*, document_text: str, retrieved_context: str) -> str:
    schema = claim_json_schema()

    return (
        "You extract structured data from messy auto-insurance claim documents.\n"
        "Use only information explicitly present in the document.\n"
        "Do not guess or invent values.\n"
        "If a field is not clearly stated, use null.\n"
        "\n"
        "Insured vs claimant rules:\n"
        "- insured_name is the policyholder / customer / our insured / Insd.\n"
        "- claimant_name is the person making the claim. In many first-party claim documents it is the same as insured.\n"
        "- For estimates and repair orders, the customer name is typically the insured.\n"
        "- For support tickets and emails, the sender is typically the insured.\n"
        "- If the document mentions only a single person and no third-party claimant is present, set claimant_name equal to insured_name.\n"
        "Return only valid JSON that matches the provided JSON Schema.\n"
        "Dates should be YYYY-MM-DD if possible.\n"
        "loss_description must be a compact phrase list separated by semicolons (no full sentences), without adding facts.\n"
        "coverage_type must be one of: collision, comprehensive, liability, unknown.\n"
        "\n"
        "JSON Schema:\n"
        f"{schema}\n"
        "\n"
        "Reference context (glossary/policy snippets):\n"
        f"{retrieved_context}\n"
        "\n"
        "Document:\n"
        f"{document_text}\n"
    )


def build_fix_json_prompt(*, schema_json: str, bad_json: str, error: str) -> str:
    return (
        "Fix the JSON so it is valid and matches the JSON Schema.\n"
        "Return only the corrected JSON.\n"
        "\n"
        "JSON Schema:\n"
        f"{schema_json}\n"
        "\n"
        "Invalid JSON:\n"
        f"{bad_json}\n"
        "\n"
        "Error:\n"
        f"{error}\n"
    )
