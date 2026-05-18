"""Guard generic CUA-framework prompts against vertical-specific contamination.

The recipe overlay machinery (``recipes/marketplace_listings/*``,
``SiteConfig.overlay``) exists to carry vertical concerns; generic prompts
must stay neutral so the framework handles arbitrary apps. This test
scans generic prompt surfaces and fails if any token from the denylist
re-enters.

Coverage grows incrementally with the PR chain that cleans up
contamination (issues #460-#464). The denylist is the union of all
tokens we never want in any generic prompt; the parametrized prompt
list grows per PR as each surface is cleaned:

- PR-1 (#460) — recovery_analysis.txt
- PR-2 (#461) — plan_decomposer.DECOMPOSE_PROMPT
- PR-4 (#464) — graph.{learner,probe,enhancer} prompts
"""

from __future__ import annotations

from pathlib import Path

import pytest

_RECOVERY_TXT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "mantis_agent"
    / "prompts"
    / "files"
    / "recovery_analysis.txt"
)


# Tokens cited verbatim in the #460-#464 issue bodies. Narrow on purpose —
# we want zero false positives on legitimate generic prose ("listing item"
# in a handler docstring is fine; "BoatTrader" is not).
VERTICAL_TOKEN_DENYLIST: tuple[str, ...] = (
    # Staff CRM specifics (issue #460, #461)
    "Estimated Deal Value",
    "Robot Information",
    "Update Lead",
    "/leads?",
    "/leads/",
    # marketplace_listings / BoatTrader specifics (issue #463, #464)
    "BoatTrader",
    "MarineMax",
    "Grady-White",
    "Bayliner",
    "Tige Boats",
    "boat photo",
    "Private Seller",
    "by-owner",
    # generic suspect phrases (issue #463)
    "dealer signals",
)


def _generic_prompts() -> list[tuple[str, str]]:
    """Return (label, body) pairs for every generic-framework prompt
    that the cleanup chain has reached so far.

    Add an entry here in the PR that scrubs that surface:
    - PR-2 will add ``plan_decomposer.DECOMPOSE_PROMPT``
    - PR-4 will add the three ``graph.*`` prompts
    """
    return [
        ("prompts/files/recovery_analysis.txt", _RECOVERY_TXT.read_text()),
    ]


@pytest.mark.parametrize(
    "label,body",
    _generic_prompts(),
    ids=[label for label, _ in _generic_prompts()],
)
def test_generic_prompt_has_no_vertical_tokens(label: str, body: str) -> None:
    leaks = [tok for tok in VERTICAL_TOKEN_DENYLIST if tok in body]
    assert not leaks, (
        f"Vertical-specific tokens leaked into generic prompt {label!r}: "
        f"{leaks}. Move the example into the relevant recipe (e.g. "
        f"recipes/marketplace_listings/) or use neutral placeholders."
    )


def test_recovery_analysis_txt_exists() -> None:
    # Sanity: catches relocation of the prompt file.
    assert _RECOVERY_TXT.is_file(), f"recovery_analysis.txt missing at {_RECOVERY_TXT}"
