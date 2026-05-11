"""Tests for ``_build_oxylabs_username`` username format.

Oxylabs docs require US states to be prefixed with the country
code (``us_california``, not ``california``). The bug surfaced when
``diagnose_proxy`` reported the geo-pinned curl returning Ukraine /
Brazil / France IPs — Oxylabs was treating malformed ``st-florida``
as unknown and silently falling back to random global rotation.

Canonical Oxylabs example (from their docs):
    ``customer-USERNAME-cc-US-st-us_california-city-los_angeles``

These tests pin the format the helper produces, including the
``us_<state>`` auto-prefix when the operator writes the unprefixed
state name. Codebase has no existing tests for this helper — the
bug-fix ships its own coverage.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_helper():
    """Import ``_build_oxylabs_username`` from the Modal deploy
    module without triggering Modal CLI initialization. The deploy
    module is named ``deploy.modal.modal_plan_runner`` but isn't a
    package; load it directly by path."""
    path = Path(__file__).parent.parent / "deploy" / "modal" / "modal_plan_runner.py"
    spec = importlib.util.spec_from_file_location("_test_modal_plan_runner", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_test_modal_plan_runner"] = mod
    spec.loader.exec_module(mod)
    return mod._build_oxylabs_username


def _set_env(monkeypatch, **kwargs):
    """Set OXYLABS_* env vars for the test."""
    for k, v in kwargs.items():
        if v is None:
            monkeypatch.delenv(f"OXYLABS_{k.upper()}", raising=False)
        else:
            monkeypatch.setenv(f"OXYLABS_{k.upper()}", v)


def test_no_username_returns_empty(monkeypatch) -> None:
    monkeypatch.delenv("OXYLABS_USERNAME", raising=False)
    fn = _load_helper()
    assert fn(session="s") == ""


def test_bare_username_with_session(monkeypatch) -> None:
    """Without ``geo=True``, no cc/st/city slots are appended."""
    _set_env(monkeypatch, username="acme", country="US", state="florida", city="miami")
    fn = _load_helper()
    assert fn(session="run42", geo=False) == "customer-acme-sessid-run42"


def test_geo_country_only(monkeypatch) -> None:
    """Country-only pinning matches Oxylabs canonical
    ``customer-USERNAME-cc-DE`` form."""
    _set_env(monkeypatch, username="acme", country="DE", state=None, city=None)
    fn = _load_helper()
    assert fn(session="r", geo=True) == "customer-acme-cc-DE-sessid-r"


def test_geo_country_and_unprefixed_state_gets_us_prefix(monkeypatch) -> None:
    """The bug being fixed: ``OXYLABS_STATE=florida`` must produce
    ``st-us_florida`` (not the malformed ``st-florida`` that
    Oxylabs silently rejects)."""
    _set_env(monkeypatch, username="acme", country="US", state="florida", city=None)
    fn = _load_helper()
    out = fn(session="r", geo=True)
    assert out == "customer-acme-cc-US-st-us_florida-sessid-r"


def test_geo_preserves_prefixed_state(monkeypatch) -> None:
    """If the operator already wrote ``us_california`` (canonical
    form), don't double-prefix it."""
    _set_env(monkeypatch, username="acme", country="US", state="us_california", city=None)
    fn = _load_helper()
    out = fn(session="r", geo=True)
    assert "us_us_california" not in out
    assert "st-us_california" in out


def test_geo_state_lowercased(monkeypatch) -> None:
    """Operators may write the state with any casing; we emit
    lowercase per Oxylabs case-insensitive convention."""
    _set_env(monkeypatch, username="acme", country="US", state="Florida", city=None)
    fn = _load_helper()
    out = fn(session="r", geo=True)
    assert "st-us_florida" in out


def test_geo_city_with_state_canonical_form(monkeypatch) -> None:
    """Canonical Oxylabs combined-pin example:
    ``st-us_california-city-los_angeles``. Spaces in city → underscores."""
    _set_env(monkeypatch, username="acme", country="US",
             state="california", city="Los Angeles")
    fn = _load_helper()
    out = fn(session="r", geo=True)
    assert out == "customer-acme-cc-US-st-us_california-city-los_angeles-sessid-r"


def test_geo_city_alone_no_state(monkeypatch) -> None:
    """``cc-GB-city-london`` — city without state still works,
    matches the docs' London example."""
    _set_env(monkeypatch, username="acme", country="GB", state=None, city="london")
    fn = _load_helper()
    out = fn(session="r", geo=True)
    assert out == "customer-acme-cc-GB-city-london-sessid-r"


def test_geo_no_country_falls_back_to_bare_state(monkeypatch) -> None:
    """If somehow OXYLABS_COUNTRY is unset but state is set, don't
    invent a country prefix — emit the state as-is (lowercased).
    Edge case; operators should always set country, but defend in
    depth."""
    _set_env(monkeypatch, username="acme", country=None, state="florida", city=None)
    fn = _load_helper()
    out = fn(session="r", geo=True)
    # No "cc-" segment; state emitted as raw lowercase (no auto-prefix
    # because there's no country code to attach).
    assert "cc-" not in out
    assert "st-florida" in out
