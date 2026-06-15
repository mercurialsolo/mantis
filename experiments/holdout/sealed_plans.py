"""Hand-authored micro-plans for the sealed holdout tasks (#894).

Each entry maps a holdout `oracle_task_id` to the env it runs in + a
pre-decomposed micro-plan (the deterministic, guard-carrying path — no LLM
decompose at run time). Steps use ``{env_url}`` as a placeholder the runner
substitutes with the live Daytona preview URL.

Grounding source: each env's ``scripts/smoke.py`` (exact endpoints/fields) +
``app/templates`` (labels/buttons) + ``app/oracles/<task>.py`` (pass criteria).
The plan's job is to reach a state the env oracle grades as ``passed`` while
exercising the real UI; the producer emits a `mark_for_eval` candidate keyed
``<domain>.<plan_name>.v1`` on plan-completion (oracle grade is read separately).

The ``domain`` here becomes the Augur task_spec_id prefix, so we set it to the
holdout env id (e.g. ``indeed``) and ``plan_name`` to the oracle_task_id — giving
``indeed.t01_search_save_remote.v1``, aligned with the holdout manifest anchor.
"""

from __future__ import annotations

from typing import Any

# Daytona sandbox ids for the running sealed envs (live-identified 2026-06-14).
SANDBOXES: dict[str, str] = {
    "indeed": "a72a3ffc-f1c1-4628-89a6-1c560703441b",
    "mercor": "f0f3ffc9-4879-48a9-98db-b5b5224fe20f",
    "linkedin": "f43c50dd-0edb-4667-8b50-2ff2f697de24",
}


def _nav(url: str, intent: str, *, wait: int = 6) -> dict[str, Any]:
    return {
        "type": "navigate",
        "intent": intent,
        "params": {"url": url, "wait_after_load_seconds": wait},
        "required": True,
        "section": "setup",
        "gate": False,
    }


def _click(intent: str, *, label: str = "", required: bool = True) -> dict[str, Any]:
    """A plain element click (link/button that navigates or opens a modal).

    The deployed click/form handlers key on ``label`` (NOT ``target``) — a
    ``target`` param reaches the form-submit path as an empty label and fails
    with ``button '' not found`` (live-diagnosed 2026-06-14)."""
    params: dict[str, Any] = {}
    if label:
        params["label"] = label
    return {
        "type": "click",
        "intent": intent,
        "params": params,
        "required": required,
        "section": "act",
        "gate": False,
    }


def _submit(label: str, intent: str, *, required: bool = True) -> dict[str, Any]:
    """A form-submit button click (e.g. 'Save job', 'Post', 'Mark reviewed').

    Uses step type ``submit`` which the form handler resolves via
    ``find_form_target`` against the ``label``."""
    return {
        "type": "submit",
        "intent": intent,
        "params": {"label": label},
        "required": required,
        "section": "act",
        "gate": False,
    }


def _fill(label: str, value: str, intent: str) -> dict[str, Any]:
    return {
        "type": "fill_field",
        "intent": intent,
        "params": {"label": label, "value": value},
        "required": True,
        "section": "act",
        "gate": False,
    }


# ── the sealed task registry ───────────────────────────────────────
# Each: env, plan_name (== oracle_task_id), oracle_task_id, steps.

SEALED_TASKS: dict[str, dict[str, Any]] = {
    # indeed t01 — filtered search (remote) fires the search audit on results
    # load; saving job_00007 (the remote Austin SE) records the job_saved row.
    "indeed.t01_search_save_remote": {
        "env": "indeed",
        "plan_name": "t01_search_save_remote",
        "task_text": "Search 'software engineer' in 'Austin, TX' with remote on, and save job_00007.",
        "oracle_task_id": "t01_search_save_remote",
        "steps": [
            _nav(
                "{env_url}/jobs?q=software+engineer&l=Austin&remote=1&vjk=0000000000000007",
                "Open the Indeed results for 'software engineer' in 'Austin' with Remote on, with the Software Engineer (Acme Software) job pre-selected in the detail pane",
            ),
            _submit(
                "Save job",
                "Click the 'Save job' button in the right-hand detail pane to save the selected Software Engineer job",
                required=False,  # form-POST returns JSON; oracle arbitrates the save
            ),
        ],
    },
    # indeed t03 — employer moves the seeded new applicant on job_00003 to
    # 'reviewed' via the "Mark reviewed" button on the posting page.
    "indeed.t03_employer_review_applicant": {
        "env": "indeed",
        "plan_name": "t03_employer_review_applicant",
        "task_text": "On the employer posting for job_00003, move the new applicant to 'reviewed'.",
        "oracle_task_id": "t03_employer_review_applicant",
        "steps": [
            _nav(
                "{env_url}/employers/jobs/job_00003",
                "Open the employer posting page for job_00003 to see its applicants",
            ),
            _submit(
                "Mark reviewed",
                "Click 'Mark reviewed' on the new applicant to move them from 'new' to 'reviewed'",
                required=False,
            ),
        ],
    },
    # linkedin t02 — open the "Start a post" composer, write text with a
    # #hashtag, and Post. extract_hashtags fires on the /feed/post route.
    "linkedin.t02_post_text_update": {
        "env": "linkedin",
        "plan_name": "t02_post_text_update",
        "task_text": "Create a LinkedIn feed post containing text and at least one #hashtag.",
        "oracle_task_id": "t02_post_text_update",
        "steps": [
            _nav("{env_url}/feed/", "Open the LinkedIn home feed"),
            _click(
                "Click the 'Start a post' button to open the post composer",
                label="Start a post",
                required=False,  # opens a JS <dialog> (showModal) → no nav → runner sees no_state_change
            ),
            _fill(
                "What do you want to talk about?",
                "Thrilled to be building computer-use agents this quarter. #buildinpublic",
                "Type a short update that includes a #hashtag into the post composer text box",
            ),
            _submit("Post", "Click the 'Post' button to publish the update", required=False),
        ],
    },
}
