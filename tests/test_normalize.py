from auto_claim_rag.extraction.normalize import (
    location_city_state,
    normalize_date,
    normalize_email,
    normalize_location,
    normalize_loss_description,
    normalize_phone,
    normalize_person_name,
    normalize_police_report_number,
    normalize_policy_number,
    parse_money_usd,
)


def test_normalize_email() -> None:
    assert normalize_email("  Foo@Example.Com ") == "foo@example.com"
    assert normalize_email("") is None
    assert normalize_email(None) is None


def test_normalize_phone() -> None:
    assert normalize_phone("(206) 555-0192") == "206-555-0192"
    assert normalize_phone("206.555.0192") == "206-555-0192"
    assert normalize_phone("+1 (206) 555-0192") == "12065550192"
    assert normalize_phone(None) is None


def test_parse_money_usd() -> None:
    assert parse_money_usd("$3,878.60") == 3878.60
    assert parse_money_usd("3,878.60") == 3878.60
    assert parse_money_usd("3878.60") == 3878.60
    assert parse_money_usd("3878") == 3878.0
    assert parse_money_usd("2.45k") == 2450.0
    assert parse_money_usd(None) is None


def test_normalize_date() -> None:
    assert normalize_date("2026/06/18") == "2026-06-18"
    assert normalize_date("2026-06-18") == "2026-06-18"
    assert normalize_date(None) is None


def test_normalize_person_name() -> None:
    assert normalize_person_name("Adebayo, Jordan") == "Jordan Adebayo"
    assert normalize_person_name("  Jordan   Adebayo ") == "Jordan Adebayo"


def test_normalize_policy_number() -> None:
    assert normalize_policy_number("NYP 33110 08") == "NYP-33110-08"
    assert normalize_policy_number("AZP-44109-19") == "AZP-44109-19"


def test_normalize_police_report_number() -> None:
    assert normalize_police_report_number("MBPD case 26-0705-9921") == "26-0705-9921"


def test_normalize_location() -> None:
    assert normalize_location("near 55 W 125th St, New York, NY") == "55 W 125th St, New York, NY"
    assert normalize_location("I-35 near Exit 247, Austin TX (northbound)") == "I-35 near Exit 247, Austin TX"
    assert location_city_state("4550 Aurora Ave N, Seattle, WA") == "Seattle, WA"


def test_normalize_loss_description() -> None:
    assert (
        normalize_loss_description(
            "The insured was turning left. The car is drivable but has bumper damage."
        )
        == "was turning left; car is drivable but has bumper damage"
    )
