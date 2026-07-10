from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from auto_claim_rag.embeddings import Embedder
from auto_claim_rag.extraction.llm import LLM
from auto_claim_rag.extraction.normalize import (
    normalize_date,
    normalize_email,
    location_city_state,
    normalize_location,
    normalize_loss_description,
    normalize_phone,
    normalize_person_name,
    normalize_police_report_number,
    normalize_policy_number,
    parse_money_usd,
)
from auto_claim_rag.extraction.prompts import (
    build_claim_extraction_prompt,
    build_fix_json_prompt,
    claim_json_schema,
)
from auto_claim_rag.extraction.schema import ClaimExtraction
from auto_claim_rag.retrieval import retrieve
from auto_claim_rag.vectorstore import VectorIndex


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _coalesce_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    s = value.strip()
    return s if s else None


def _fallback_claim_id(document_text: str) -> str | None:
    m = re.search(r"\bCLM#?:?\s*([A-Za-z0-9-]{6,30})\b", document_text, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"\bClaim(?: number| id)?:\s*([A-Za-z0-9-]{6,30})\b", document_text, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _fallback_policy_number(document_text: str) -> str | None:
    m = re.search(r"\bPOL#?:?\s*([A-Za-z0-9 -]{4,30})", document_text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"\bPolicy(?: number)?:\s*([A-Za-z0-9 -]{4,30})", document_text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _fallback_vin(document_text: str) -> str | None:
    m = re.search(r"\bVIN[:\s#]*([A-HJ-NPR-Z0-9]{17})\b", document_text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def _preprocess_extraction_dict(data: dict[str, Any]) -> dict[str, Any]:
    if "claim_id" in data:
        data["claim_id"] = normalize_policy_number(data.get("claim_id"))
    if "contact_phone" in data:
        data["contact_phone"] = normalize_phone(data.get("contact_phone"))
    if "contact_email" in data:
        data["contact_email"] = normalize_email(data.get("contact_email"))
    if "loss_location" in data:
        data["loss_location"] = normalize_location(data.get("loss_location"))
    if "loss_description" in data:
        data["loss_description"] = normalize_loss_description(data.get("loss_description"))
    if "estimated_damage_amount_usd" in data:
        data["estimated_damage_amount_usd"] = parse_money_usd(
            data.get("estimated_damage_amount_usd")
        )
    if "date_of_loss" in data:
        data["date_of_loss"] = normalize_date(data.get("date_of_loss"))
    if "policy_number" in data:
        data["policy_number"] = normalize_policy_number(data.get("policy_number"))
    if "police_report_number" in data:
        data["police_report_number"] = normalize_police_report_number(
            data.get("police_report_number")
        )
    if "insured_name" in data:
        data["insured_name"] = normalize_person_name(data.get("insured_name"))
    if "claimant_name" in data:
        data["claimant_name"] = normalize_person_name(data.get("claimant_name"))
    return data


@dataclass
class AutoClaimExtractor:
    llm: LLM
    kb_index: VectorIndex
    embedder: Embedder
    retrieve_top_k: int = 6
    max_fix_attempts: int = 2

    def _retrieve_kb_hits(self, document_text: str):
        query = (
            "Auto insurance claims glossary and coverage interpretation.\n"
            + document_text[:800]
        )
        return retrieve(
            index=self.kb_index,
            embedder=self.embedder,
            query=query,
            top_k=self.retrieve_top_k,
        )

    def _retrieve_kb_context(self, document_text: str) -> str:
        hits = self._retrieve_kb_hits(document_text)
        lines: list[str] = []
        for h in hits:
            snippet = h.chunk.text.strip().replace("\n", " ")
            lines.append(f"[{h.chunk.chunk_id}] {snippet}")
        return "\n".join(lines)

    def extract_with_trace(self, document_text: str):
        hits = self._retrieve_kb_hits(document_text)
        context_lines: list[str] = []
        retrieved = []
        for h in hits:
            snippet = h.chunk.text.strip().replace("\n", " ")
            context_lines.append(f"[{h.chunk.chunk_id}] {snippet}")
            retrieved.append({"chunk_id": h.chunk.chunk_id, "score": h.score})

        extracted = self._extract_from_context(
            document_text=document_text, retrieved_context="\n".join(context_lines)
        )
        return extracted, retrieved

    def extract(self, *, document_text: str) -> ClaimExtraction:
        context = self._retrieve_kb_context(document_text)
        return self._extract_from_context(
            document_text=document_text, retrieved_context=context
        )

    def _extract_from_context(self, *, document_text: str, retrieved_context: str) -> ClaimExtraction:
        prompt = build_claim_extraction_prompt(
            document_text=document_text, retrieved_context=retrieved_context
        )
        raw = _strip_code_fences(self.llm.complete(prompt))

        schema = claim_json_schema()

        attempt = 0
        last_error: str | None = None
        while True:
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    data = _preprocess_extraction_dict(data)

                    if _coalesce_str(data.get("claim_id")) is None:
                        cid = _fallback_claim_id(document_text)
                        if cid:
                            data["claim_id"] = normalize_policy_number(cid)

                    if _coalesce_str(data.get("policy_number")) is None:
                        pol = _fallback_policy_number(document_text)
                        if pol:
                            data["policy_number"] = normalize_policy_number(pol)

                    if _coalesce_str(data.get("vehicle_vin")) is None:
                        vin = _fallback_vin(document_text)
                        if vin:
                            data["vehicle_vin"] = vin

                extracted = ClaimExtraction.model_validate(data)

                if (
                    extracted.claimant_name is None
                    and extracted.insured_name is not None
                    and not re.search(r"\bclaimant\b|\bclmt\b", document_text, flags=re.IGNORECASE)
                    and not re.search(r"\bour insured\b", document_text, flags=re.IGNORECASE)
                ):
                    extracted = extracted.model_copy(update={"claimant_name": extracted.insured_name})

                loss_loc = normalize_location(extracted.loss_location)
                if (
                    loss_loc is not None
                    and re.search(r"\bestimate\b|\bauto body\b|\bro\b", document_text, flags=re.IGNORECASE)
                    and re.search(r"\baddress\b", document_text, flags=re.IGNORECASE)
                ):
                    loss_loc = location_city_state(loss_loc)

                extracted = extracted.model_copy(
                    update={
                        "claim_id": normalize_policy_number(extracted.claim_id),
                        "policy_number": normalize_policy_number(extracted.policy_number),
                        "insured_name": normalize_person_name(extracted.insured_name),
                        "claimant_name": normalize_person_name(extracted.claimant_name),
                        "contact_phone": normalize_phone(extracted.contact_phone),
                        "contact_email": normalize_email(extracted.contact_email),
                        "loss_location": loss_loc,
                        "loss_description": normalize_loss_description(extracted.loss_description),
                        "date_of_loss": normalize_date(extracted.date_of_loss),
                        "police_report_number": normalize_police_report_number(
                            extracted.police_report_number
                        ),
                        "estimated_damage_amount_usd": parse_money_usd(
                            extracted.estimated_damage_amount_usd
                        ),
                    }
                )
                return extracted
            except Exception as e:
                last_error = str(e)
                if attempt >= self.max_fix_attempts:
                    raise ValueError(
                        "Failed to produce valid structured output: " + (last_error or "")
                    )

                fix_prompt = build_fix_json_prompt(
                    schema_json=schema, bad_json=raw, error=last_error
                )
                raw = _strip_code_fences(self.llm.complete(fix_prompt))
                attempt += 1
