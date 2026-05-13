"""Tests for plan URL templating — ``{{ENV_URL}}`` substitution.

The harness substitutes ``{{ENV_URL}}`` for the booted env's base URL
on every string-shaped field in the plan JSON. These tests assert:

* String fields under ``params.url`` get substituted.
* Free-text intents (e.g. ``"Navigate to {{ENV_URL}}/foo"``) also get
  substituted — common when authors write plans by hand.
* Non-string values are untouched.
* Trailing slashes on ``env_url`` are normalised so we don't end up with
  ``http://host//path``.
"""

from __future__ import annotations

from mantis_agent.sim_envs.templating import (
    ENV_URL_PLACEHOLDER,
    substitute_env_url,
)


def test_substitutes_params_url():
    plan = {
        "steps": [
            {
                "type": "navigate",
                "params": {"url": f"{ENV_URL_PLACEHOLDER}/contacts"},
                "intent": "Open contacts",
            }
        ],
    }
    out = substitute_env_url(plan, "http://127.0.0.1:8001")
    assert out["steps"][0]["params"]["url"] == "http://127.0.0.1:8001/contacts"


def test_substitutes_intent_string():
    plan = {
        "steps": [
            {
                "type": "navigate",
                "intent": f"Open {ENV_URL_PLACEHOLDER}/deals",
                "params": {},
            }
        ],
    }
    out = substitute_env_url(plan, "http://example/")
    assert out["steps"][0]["intent"] == "Open http://example/deals"


def test_normalises_trailing_slash():
    plan = {"steps": [{"params": {"url": f"{ENV_URL_PLACEHOLDER}/x"}}]}
    out = substitute_env_url(plan, "http://h:1234///")
    assert out["steps"][0]["params"]["url"] == "http://h:1234/x"


def test_leaves_non_string_fields_untouched():
    plan = {
        "steps": [
            {
                "type": "loop",
                "loop_count": 5,
                "params": {"flags": [True, False, 1]},
            }
        ],
    }
    out = substitute_env_url(plan, "http://x")
    assert out["steps"][0]["loop_count"] == 5
    assert out["steps"][0]["params"]["flags"] == [True, False, 1]


def test_no_placeholder_is_passthrough():
    plan = {"steps": [{"params": {"url": "http://live.example/page"}}]}
    out = substitute_env_url(plan, "http://anything")
    assert out["steps"][0]["params"]["url"] == "http://live.example/page"


def test_nested_dict_in_params():
    plan = {
        "steps": [
            {
                "params": {
                    "nested": {
                        "deeper": {"url": f"{ENV_URL_PLACEHOLDER}/q"},
                    },
                },
            }
        ],
    }
    out = substitute_env_url(plan, "http://e")
    assert out["steps"][0]["params"]["nested"]["deeper"]["url"] == "http://e/q"
