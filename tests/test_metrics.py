from auto_claim_rag.eval.metrics import score_extraction
from auto_claim_rag.extraction.schema import ClaimExtraction


def test_string_normalization_basic() -> None:
    pred = ClaimExtraction(vehicle_make=" Toyota ")
    gold = ClaimExtraction(vehicle_make="toyota")
    scored = score_extraction(doc_id="x", pred=pred, gold=gold)
    assert scored.exact_match is True


def test_float_tolerance() -> None:
    pred = ClaimExtraction(estimated_damage_amount_usd=100.5)
    gold = ClaimExtraction(estimated_damage_amount_usd=100.0)
    scored = score_extraction(doc_id="x", pred=pred, gold=gold)
    assert scored.exact_match is True

    pred2 = ClaimExtraction(estimated_damage_amount_usd=102.1)
    gold2 = ClaimExtraction(estimated_damage_amount_usd=100.0)
    scored2 = score_extraction(doc_id="x", pred=pred2, gold=gold2)
    assert scored2.exact_match is False


def test_phone_digits_only_match() -> None:
    pred = ClaimExtraction(contact_phone="(206) 555-0192")
    gold = ClaimExtraction(contact_phone="206-555-0192")
    scored = score_extraction(doc_id="x", pred=pred, gold=gold)
    assert scored.exact_match is True


def test_loss_description_similarity_match() -> None:
    pairs = [
        (
            "Turning left; SUV ran red; clipped front passenger side; bumper/fender damage; drivable",
            "Turning left, SUV ran red light, clipped front passenger side; bumper/fender damage; drivable.",
        ),
        (
            "hit tire tread in roadway; struck guardrail; tow required; vehicle not drivable",
            "Hit tire tread in roadway then struck guardrail; tow required; vehicle not drivable.",
        ),
        (
            "hail storm; dents on hood; dents on roof; no injuries",
            "Hail storm caused dents on hood and roof; no injuries; estimate from River City Dent Repair.",
        ),
    ]

    for pred_desc, gold_desc in pairs:
        pred = ClaimExtraction(loss_description=pred_desc)
        gold = ClaimExtraction(loss_description=gold_desc)
        scored = score_extraction(doc_id="x", pred=pred, gold=gold)
        assert scored.exact_match is True
