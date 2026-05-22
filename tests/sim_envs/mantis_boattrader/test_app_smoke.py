"""End-to-end smoke against the FastAPI app via httpx TestClient.

Covers the oracle plumbing shipped in issue #588:

* Harness gating — ``/__env__/*`` (except ``/health``) requires
  ``X-Env-Admin``.
* ``/__env__/health`` is open.
* ``/__env__/oracle?task_id=<id>`` returns the canonical
  "no oracle registered" shape for an unknown task.
* Public mutating routes (`/__site/consent`, ``/boat/<slug>/contact``,
  ``/boat/<slug>/show-phone``) populate the mutations audit log.
* ``/__env__/state`` surfaces the mutation count.
"""

from __future__ import annotations

import pytest


pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("jinja2")
pytest.importorskip("multipart")  # python-multipart is needed for Form parsing


@pytest.fixture
def client():
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from app.main import create_app  # noqa: PLC0415

    app = create_app()
    with TestClient(app) as c:
        yield c


def _admin_headers():
    return {"X-Env-Admin": "test-admin-token"}


# ── harness gating ─────────────────────────────────────────────────────


@pytest.mark.parametrize("path", [
    "/__env__/state",
    "/__env__/leads",
    "/__env__/mutations",
    "/__env__/oracle?task_id=DOES_NOT_EXIST",
])
def test_admin_routes_403_without_token(client, path):
    r = client.get(path)
    assert r.status_code == 403


def test_health_is_open(client):
    r = client.get("/__env__/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["boats"] > 0


# ── oracle endpoint ────────────────────────────────────────────────────


def test_oracle_unknown_task_id_returns_no_grader(client):
    r = client.get(
        "/__env__/oracle?task_id=DOES_NOT_EXIST",
        headers=_admin_headers(),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["passed"] is False
    assert body["score"] == 0.0
    assert body["task_id"] == "DOES_NOT_EXIST"
    assert "no oracle" in " ".join(body["reasons"]).lower()
    assert body["diff"] == {}


def test_oracle_missing_task_id_returns_helpful_error(client):
    r = client.get("/__env__/oracle", headers=_admin_headers())
    assert r.status_code == 200
    body = r.json()
    assert body["passed"] is False
    assert "task_id is required" in " ".join(body["reasons"])


# ── mutations audit log ────────────────────────────────────────────────


def test_state_reports_zero_mutations_on_fresh_boot(client):
    r = client.get("/__env__/state", headers=_admin_headers())
    assert r.status_code == 200
    body = r.json()
    assert body["mutations"] == 0
    assert body["recent_mutations"] == []


def test_consent_post_emits_mutation(client):
    r = client.post(
        "/__site/consent",
        data={"choice": "accept", "next_url": "/"},
        follow_redirects=False,
    )
    assert r.status_code in (200, 303)
    state = client.get("/__env__/state", headers=_admin_headers()).json()
    assert state["mutations"] == 1
    last = state["recent_mutations"][-1]
    assert last["operation"] == "consent_set"
    assert last["payload"]["choice"] == "accept"


def test_lead_submission_emits_mutation_with_boat_id(client):
    # Pick a real boat slug via the listings page; we don't care which.
    boats = client.get("/__env__/state", headers=_admin_headers()).json()
    assert boats["boats"] > 0

    # Hit one detail page to discover a valid slug. The home page
    # references real boats in its featured rail.
    home = client.get("/")
    assert home.status_code == 200

    # Use the leads dump to confirm zero baseline.
    leads0 = client.get("/__env__/leads", headers=_admin_headers()).json()
    assert leads0["leads"] == []

    # Find a slug from the harness state's facets is not provided; pick
    # any boat via the listings API. The boats page has 24 per page.
    r = client.get("/boats/?per_page=1")
    assert r.status_code == 200
    # The HTML response embeds /boat/<slug>/ links — extract one.
    import re
    m = re.search(r'href="/boat/([^/]+)/"', r.text)
    assert m, "expected at least one /boat/<slug>/ link on the listings page"
    slug = m.group(1)

    r = client.post(
        f"/boat/{slug}/contact",
        data={
            "name": "Test Buyer",
            "email": "buyer@example.test",
            "phone": "",
            "message": "Interested.",
        },
        follow_redirects=False,
    )
    assert r.status_code == 200

    leads_after = client.get("/__env__/leads", headers=_admin_headers()).json()
    assert len(leads_after["leads"]) == 1

    muts = client.get(
        "/__env__/mutations",
        headers=_admin_headers(),
    ).json()
    lead_muts = [m for m in muts["mutations"] if m["operation"] == "lead_submitted"]
    assert len(lead_muts) == 1
    assert lead_muts[0]["target_type"] == "boat"
    assert lead_muts[0]["payload"]["email"] == "buyer@example.test"


def test_show_phone_emits_phone_revealed_mutation(client):
    """At least one boat in seed has an owner phone; if not, the test
    asserts a clean 404 path instead of a false positive."""
    # Discover any slug from the listing page.
    import re
    r = client.get("/boats/")
    assert r.status_code == 200
    slugs = re.findall(r'href="/boat/([^/]+)/"', r.text)
    assert slugs, "expected at least one boat slug"

    posted = 0
    for slug in slugs[:10]:
        r = client.post(f"/boat/{slug}/show-phone", follow_redirects=False)
        if r.status_code == 303:
            posted += 1
            break
        # 404 here means this boat has no owner phone — try the next.

    state = client.get("/__env__/state", headers=_admin_headers()).json()
    mut_ops = [m["operation"] for m in state["recent_mutations"]]
    if posted:
        assert "phone_revealed" in mut_ops
    else:
        # No private-seller boats in the seed slice we sampled — the
        # plumbing is exercised by the consent and lead tests above.
        assert "phone_revealed" not in mut_ops


# ── reset clears state ─────────────────────────────────────────────────


# ── BT01 end-to-end via public routes ──────────────────────────────────


def test_bt01_oracle_passes_on_end_to_end_qualifying_submission(client):
    """Exercise the full agent-facing chain that BT01 grades against:
    filtered listings page → detail page → contact form submit →
    oracle returns passed=true.
    """
    # 1. Filtered listings page returns at least one matching boat.
    listings = client.get(
        "/boats/?condition=used&make=Sea+Ray&price_max=200000"
    )
    assert listings.status_code == 200

    import re
    slugs = re.findall(r'href="/boat/([^/]+)/"', listings.text)
    assert slugs, "expected at least one matching boat in the listings page"

    # 2. Pick the first matching slug; confirm the detail page loads.
    slug = slugs[0]
    detail = client.get(f"/boat/{slug}/")
    assert detail.status_code == 200

    # 3. Submit the contact form with a canonical payload.
    submit = client.post(
        f"/boat/{slug}/contact",
        data={
            "name": "Test Buyer",
            "email": "buyer@example.test",
            "phone": "",
            "message": "Interested in this boat.",
        },
        follow_redirects=False,
    )
    assert submit.status_code == 200

    # 4. Oracle should now report passed=true with score=1.0.
    oracle_resp = client.get(
        "/__env__/oracle?task_id=BT01_lead_capture_filtered_search",
        headers=_admin_headers(),
    )
    assert oracle_resp.status_code == 200
    body = oracle_resp.json()
    assert body["passed"] is True, body
    assert body["score"] == 1.0
    assert body["task_id"] == "BT01_lead_capture_filtered_search"
    assert body["diff"]["hits"] == 1
    assert body["diff"]["misses"] == 0


def test_bt01_oracle_fails_when_lead_lands_on_wrong_boat(client):
    """Submitting a lead on a non-qualifying boat fails BT01."""
    listings = client.get("/boats/?make=Hatteras&condition=new")
    assert listings.status_code == 200

    import re
    slugs = re.findall(r'href="/boat/([^/]+)/"', listings.text)
    assert slugs, "expected at least one Hatteras boat in the listings page"

    slug = slugs[0]
    submit = client.post(
        f"/boat/{slug}/contact",
        data={
            "name": "Test Buyer",
            "email": "buyer@example.test",
            "phone": "",
            "message": "Interested.",
        },
        follow_redirects=False,
    )
    assert submit.status_code == 200

    oracle_resp = client.get(
        "/__env__/oracle?task_id=BT01_lead_capture_filtered_search",
        headers=_admin_headers(),
    )
    assert oracle_resp.status_code == 200
    body = oracle_resp.json()
    assert body["passed"] is False
    assert body["diff"]["hits"] == 0
    assert body["diff"]["misses"] == 1


def test_reset_clears_mutations(client):
    # Seed one mutation.
    client.post(
        "/__site/consent",
        data={"choice": "accept", "next_url": "/"},
        follow_redirects=False,
    )
    state_before = client.get("/__env__/state", headers=_admin_headers()).json()
    assert state_before["mutations"] >= 1

    r = client.post("/__env__/reset", headers=_admin_headers())
    assert r.status_code == 200

    state_after = client.get("/__env__/state", headers=_admin_headers()).json()
    # ``env_reset`` is itself a mutation stamped after the wipe so the
    # boundary is observable. So we expect exactly one mutation.
    assert state_after["mutations"] == 1
    assert state_after["recent_mutations"][-1]["operation"] == "env_reset"
    assert state_after["leads"] == 0
