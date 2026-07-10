from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClaimExtraction(BaseModel):
    claim_id: str | None = Field(default=None)
    policy_number: str | None = Field(default=None)

    insured_name: str | None = Field(default=None)
    claimant_name: str | None = Field(default=None)

    date_of_loss: str | None = Field(default=None, description="YYYY-MM-DD if possible")
    loss_location: str | None = Field(default=None)

    vehicle_year: int | None = Field(default=None)
    vehicle_make: str | None = Field(default=None)
    vehicle_model: str | None = Field(default=None)
    vehicle_vin: str | None = Field(default=None)

    loss_description: str | None = Field(default=None)
    police_report_number: str | None = Field(default=None)
    injury_reported: bool | None = Field(default=None)

    estimated_damage_amount_usd: float | None = Field(default=None)
    repair_shop_name: str | None = Field(default=None)

    contact_phone: str | None = Field(default=None)
    contact_email: str | None = Field(default=None)

    coverage_type: Literal["collision", "comprehensive", "liability", "unknown"] = "unknown"
