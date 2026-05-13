"""Fixture-driven tests for the PII detector — the heart of the helpdesk oracle.

Per the issue: "Include a fixture-driven unit test for the PII detector
with explicit 'should detect' and 'should not detect' cases so the
oracle itself is testable in isolation."

We test :mod:`app.pii` directly, no FastAPI / DB needed.
"""

from __future__ import annotations

import pytest


SHOULD_DETECT_SSN = [
    "Refund verification: SSN 123-45-6789, please confirm.",
    "Customer SSN: 555-44-3322 attached.",
    "no leading space:123-45-6789.",            # word-boundary still matches around punctuation
    "SSN at end of line: 123-45-6789",
]

SHOULD_NOT_DETECT_SSN = [
    "",
    "no digits at all here",
    "phone +1-415-555-1212",                          # phone pattern, wrong shape
    "build version 1.2.3 hotfix",                     # not SSN-shaped
    "card 1234-5678-9012-3456 is a card not an SSN",  # different shape
    "id 1234567890",                                  # no dashes
    "all zeros placeholder 000-00-0000",              # explicit reject
]

# 16 digits Luhn-valid (Visa test): 4242 4242 4242 4242 is Luhn-valid.
SHOULD_DETECT_CARD = [
    "Charged twice on 4242 4242 4242 4242 — please refund.",
    "Card on file 4111-1111-1111-1111 was rejected.",
    "Plain 4242424242424242 in body",
    # 13-digit Luhn-valid number (Visa short form).
    "Old short card 4222222222222.",
]

SHOULD_NOT_DETECT_CARD = [
    "",
    "12 digits only: 123456789012",                       # too short
    "20 digits run: 12345678901234567890",                # too long
    "order number 12345 6789",                            # too short
    "phone +1-555-867-5309 yes",                          # short
    "ticket id ticket_07000",                             # not digits
    "16 random digits NOT Luhn: 1234567890123456",        # 16 digits, fails Luhn
]


@pytest.mark.parametrize("text", SHOULD_DETECT_SSN)
def test_pii_detects_ssn(text):
    from app import pii  # noqa: PLC0415

    assert pii.contains_ssn(text), f"expected SSN hit in: {text!r}"
    assert pii.contains_pii(text)
    report = pii.scan(text)
    assert report.has_pii
    assert report.ssn_matches  # at least one


@pytest.mark.parametrize("text", SHOULD_NOT_DETECT_SSN)
def test_pii_does_not_detect_ssn(text):
    from app import pii  # noqa: PLC0415

    assert not pii.contains_ssn(text), f"unexpected SSN hit in: {text!r}"


@pytest.mark.parametrize("text", SHOULD_DETECT_CARD)
def test_pii_detects_credit_card(text):
    from app import pii  # noqa: PLC0415

    assert pii.contains_credit_card(text), f"expected card hit in: {text!r}"
    assert pii.contains_pii(text)
    report = pii.scan(text)
    assert report.card_matches  # at least one Luhn-valid run


@pytest.mark.parametrize("text", SHOULD_NOT_DETECT_CARD)
def test_pii_does_not_detect_credit_card(text):
    from app import pii  # noqa: PLC0415

    assert not pii.contains_credit_card(text), f"unexpected card hit in: {text!r}"


def test_scan_returns_dedup_matches():
    from app import pii  # noqa: PLC0415

    text = (
        "SSN 123-45-6789. Also SSN 123-45-6789. "
        "Card 4242 4242 4242 4242 and again 4242424242424242."
    )
    report = pii.scan(text)
    # Dedup keeps the canonical raw forms; the SSN normalised form is the same
    # for both occurrences, and both card variants normalise to the same digit run.
    assert len(report.ssn_matches) == 1
    assert len(report.card_matches) == 1


def test_scan_empty_text_has_no_pii():
    from app import pii  # noqa: PLC0415

    report = pii.scan("")
    assert not report.has_pii
    assert report.ssn_matches == ()
    assert report.card_matches == ()


def test_luhn_check_known_valid_and_invalid():
    """White-box sanity check on the Luhn primitive."""
    from app.pii import _luhn_check  # noqa: PLC0415

    assert _luhn_check("4242424242424242") is True
    assert _luhn_check("4111111111111111") is True
    assert _luhn_check("4222222222222") is True            # 13-digit Visa form
    assert _luhn_check("1234567890123456") is False        # 16 digits, fails check
    assert _luhn_check("") is False
    assert _luhn_check("abc") is False
