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
    "fiverr": "d2c59f51-ca2e-48d2-b436-370b18673b79",
    # #920 v2-candidate envs — sandbox ids TODO: the live-verification pass
    # resolves + fills these (Daytona id, or a Modal sim-env URL for the
    # Modal-hosted envs crm/shop/shopify/auth). Empty ⇒ _daytona_env fails
    # fast at run time, by design — these tasks aren't runnable until wired.
    "auth": "",
    "boattrader": "",
    "shopify": "",
    "crm": "",
    "shop": "",
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
    # fiverr t03 — leave a 5-star review on order_00007. The review form on the
    # order page pre-checks the 5-star radio; filling the body + publishing
    # records the reviews row (stars=5) + review_submitted audit the oracle reads.
    "fiverr.t03_leave_5star_review": {
        "env": "fiverr",
        "plan_name": "t03_leave_5star_review",
        "task_text": "Leave a 5-star review on order_00007 with a short positive comment.",
        "oracle_task_id": "t03_leave_5star_review",
        "steps": [
            _nav("{env_url}/orders/order_00007", "Open the order page for order_00007 (it has a Leave a review form)"),
            _fill(
                "Share more about your experience...",
                "Outstanding work — delivered exactly as described, fast and polished. Highly recommend!",
                "Type a positive comment into the review body box (the 5-star rating is pre-selected)",
            ),
            _submit("Publish review", "Click 'Publish review' to submit the 5-star review", required=False),
        ],
    },

    # ══════════════════════════════════════════════════════════════════
    # #920 v2 CANDIDATES — grounded offline from each env's app/oracles/* +
    # app/templates/* (the canonical grounding source). status="candidate":
    # authored but NOT yet live-verified. The verification pass runs each
    # through Mantis (Daytona/Modal) until its oracle grades `passed`, fixing
    # interaction quirks (select vs text fill, AJAX no-nav, exact labels),
    # THEN freezes the survivors into mantis-holdout-v2. Excluded (need
    # runtime capabilities a static plan lacks): auth.T07_email_otp (dynamic
    # OTP read), boattrader.BT01/02/03 (need a live-discovered listing id +
    # loop/guard) — authored during the live pass.
    # ══════════════════════════════════════════════════════════════════

    # indeed t02 — multi-step Easy Apply wizard on job_00012 (jk 000000000000000c).
    "indeed.t02_easy_apply": {
        "env": "indeed", "status": "candidate",
        "plan_name": "t02_easy_apply", "oracle_task_id": "t02_easy_apply",
        "task_text": "Easy-apply to job_00012 with phone, resume, and screening answers.",
        "steps": [
            _nav("{env_url}/jobs/000000000000000c", "Open job_00012 and its Easy apply flow"),
            _click("Click 'Easy apply' to start the application", label="Easy apply", required=False),
            _fill("Full name", "Jordan Rivera", "Type the applicant full name"),
            _fill("Email", "jordan.rivera@example.com", "Type the applicant email"),
            _fill("Phone", "(555) 123-4567", "Type the applicant phone"),
            _submit("Continue", "Advance to the resume step", required=False),
            _submit("Continue", "Keep the pre-selected resume and advance", required=False),
            _submit("Continue", "Advance past the screening questions (defaults pre-filled)", required=False),
            _submit("Submit application", "Submit the application", required=False),
        ],
    },
    # mercor t01 — multi-step apply wizard on job_00001.
    "mercor.t01_apply_to_ml_engineer": {
        "env": "mercor", "status": "candidate",
        "plan_name": "t01_apply_to_ml_engineer", "oracle_task_id": "t01_apply_to_ml_engineer",
        "task_text": "Apply to job_00001 with the screening answers.",
        "steps": [
            _nav("{env_url}/jobs/job_00001", "Open job_00001 and its apply flow"),
            _click("Click 'Apply' to start the application", label="Apply", required=False),
            _fill("Headline", "Internal Medicine Physician, 8 years", "Type the profile headline"),
            _fill("Hourly rate ($/hr)", "150", "Type the hourly rate"),
            _submit("Next", "Advance to the resume step", required=False),
            _fill("Resume text", "Board-certified Internal Medicine physician, 8+ years clinical practice with strong diagnostic reasoning.", "Type resume text"),
            _submit("Next", "Advance to the screening step", required=False),
            _fill("Are you board-eligible or board-certified in Internal Medicine?", "Yes, board-certified", "Answer the board-certification question"),
            _fill("What is your earliest start date?", "2026-07-01", "Answer the start-date question"),
            _submit("Next", "Advance to review", required=False),
            _submit("Submit application", "Submit the application", required=False),
        ],
    },

    # auth T01 — password login.
    "auth.T01_password_login": {
        "env": "auth", "status": "candidate",
        "plan_name": "T01_password_login", "oracle_task_id": "T01_password_login",
        "task_text": "Sign in with the given username and password.",
        "steps": [
            _nav("{env_url}/login", "Open the sign-in page"),
            _fill("Email", "ada@mantis.example", "Type the account email"),
            _fill("Password", "hunter2", "Type the account password"),
            _submit("Sign in", "Submit the sign-in form", required=False),
        ],
    },
    # auth T08 — passkey assertion (simulated as a form POST; click the enrolled passkey).
    "auth.T08_passkey": {
        "env": "auth", "status": "candidate",
        "plan_name": "T08_passkey", "oracle_task_id": "T08_passkey",
        "task_text": "Complete the passkey assertion ceremony to sign in.",
        "steps": [
            _nav("{env_url}/login", "Open the sign-in page"),
            _click("Choose passkey sign-in", label="Use a passkey", required=False),
            _submit("Assert with the enrolled passkey (MacBook Touch ID)", "Submit the passkey assertion", required=False),
        ],
    },
    # auth T02 — Google OAuth (simulated picker + consent).
    "auth.T02_oauth_google": {
        "env": "auth", "status": "candidate",
        "plan_name": "T02_oauth_google", "oracle_task_id": "T02_oauth_google",
        "task_text": "Authorize sign-in via the Google OAuth provider.",
        "steps": [
            _nav("{env_url}/login", "Open the sign-in page"),
            _click("Start Google OAuth", label="Continue with Google", required=False),
            _click("Pick the seeded Google account", label="ada@mantis.example", required=False),
            _submit("Allow", "Grant consent to complete the OAuth sign-in", required=False),
        ],
    },

    # shopify t04 — create a support ticket.
    "shopify.t04_create_support_ticket": {
        "env": "shopify", "status": "candidate",
        "plan_name": "t04_create_support_ticket", "oracle_task_id": "t04_create_support_ticket",
        "task_text": "Create a support ticket with a subject, category, and description.",
        "steps": [
            _nav("{env_url}/support/contact", "Open the support contact form"),
            _fill("Subject", "API payout retrieval failing", "Type the ticket subject"),
            _fill("Category", "Payouts", "Choose the ticket category"),
            _fill("Description", "Unable to retrieve payout history via the API endpoint; returns 500.", "Type the description"),
            _submit("Submit ticket", "Submit the support ticket", required=False),
        ],
    },
    # shopify t05 — update business email in Settings.
    "shopify.t05_update_business_email": {
        "env": "shopify", "status": "candidate",
        "plan_name": "t05_update_business_email", "oracle_task_id": "t05_update_business_email",
        "task_text": "Update the business email in Settings to the given address.",
        "steps": [
            _nav("{env_url}/settings", "Open Settings"),
            _fill("Business email", "ops@newcompany.example", "Type the new business email"),
            _submit("Save", "Save the contact information", required=False),
        ],
    },
    # shopify t03 — export payouts CSV (audit-logged; download, no nav).
    "shopify.t03_export_payouts_csv": {
        "env": "shopify", "status": "candidate",
        "plan_name": "t03_export_payouts_csv", "oracle_task_id": "t03_export_payouts_csv",
        "task_text": "Export the payouts list as CSV.",
        "steps": [
            _nav("{env_url}/payouts", "Open the Payouts page"),
            _click("Export the payouts as CSV", label="Export CSV", required=False),
        ],
    },
    # shopify t11 — open a store's detail page from the Stores list.
    "shopify.t11_view_store_detail": {
        "env": "shopify", "status": "candidate",
        "plan_name": "t11_view_store_detail", "oracle_task_id": "t11_view_store_detail",
        "task_text": "Open a store's detail page from the Stores list.",
        "steps": [
            _nav("{env_url}/stores", "Open the Stores list"),
            _click("Open the first store's detail page", label="EA demostore", required=False),
        ],
    },

    # crm T04 — add a meeting note to Sarah Chen (contact_00042), dated yesterday.
    "crm.T04_add_meeting_note": {
        "env": "crm", "status": "candidate",
        "plan_name": "T04_add_meeting_note", "oracle_task_id": "T04_add_meeting_note",
        "task_text": "Add a 'meeting' note to Sarah Chen dated yesterday mentioning 'discussed Q3 expansion'.",
        "steps": [
            _nav("{env_url}/contacts/contact_00042", "Open Sarah Chen's contact record"),
            _click("Open the Activity tab", label="Activity", required=False),
            _fill("Activity type", "meeting", "Choose 'Log meeting' as the activity type"),
            _fill("What happened? (notes, summary, action items…)", "discussed Q3 expansion", "Type the note body"),
            _fill("Date", "2026-06-13T00:00:00Z", "Set the activity date to yesterday"),
            _submit("Save activity", "Save the meeting note", required=False),
        ],
    },
    # crm T02 — merge the four acme dupes into contact_00001.
    "crm.T02_merge_acme_dupes": {
        "env": "crm", "status": "candidate",
        "plan_name": "T02_merge_acme_dupes", "oracle_task_id": "T02_merge_acme_dupes",
        "task_text": "Merge the four alice.lead@acme.com contacts, keeping contact_00001.",
        "steps": [
            _nav("{env_url}/contacts/contact_00001", "Open the survivor contact (contact_00001)"),
            _fill("Merge duplicate contact ids", "contact_00002, contact_00003, contact_00004", "List the loser contact ids to merge in"),
            _submit("Merge into this contact", "Merge the dupes into contact_00001", required=False),
        ],
    },

    # shop T03 — create a 20%-off coupon for outerwear-women.
    "shop.T03_create_coupon": {
        "env": "shop", "status": "candidate",
        "plan_name": "T03_create_coupon", "oracle_task_id": "T03_create_coupon",
        "task_text": "Create a 20%-off coupon for outerwear-women, expiring 2026-02-15, max 100 uses.",
        "steps": [
            _nav("{env_url}/admin/coupons", "Open the coupons admin"),
            _fill("Code", "WOMENS20", "Type a coupon code"),
            _fill("Type", "pct", "Choose '% off'"),
            _fill("value (20 for 20%, 5 for $5)", "20", "Set 20% off"),
            _fill("Scope category", "outerwear-women", "Scope to outerwear-women"),
            _fill("expires", "2026-02-15", "Set the expiry date"),
            _fill("max uses", "100", "Set the max uses"),
            _submit("Create", "Create the coupon", required=False),
        ],
    },
    # shop T02 — refund line item 2 of order_04421, reason 'damaged', notify customer.
    "shop.T02_refund_line_item": {
        "env": "shop", "status": "candidate",
        "plan_name": "T02_refund_line_item", "oracle_task_id": "T02_refund_line_item",
        "task_text": "Refund line item 2 of order_04421 with reason 'damaged' and notify the customer.",
        "steps": [
            _nav("{env_url}/admin/orders/order_04421", "Open order_04421"),
            _click("Open the Refunds tab", label="Refunds", required=False),
            _fill("line", "2", "Target line item 2"),
            _fill("Reason (e.g. damaged)", "damaged", "Set the refund reason"),
            _click("Tick notify customer", label="Notify customer", required=False),
            _submit("Refund", "Submit the refund", required=False),
        ],
    },
    # shop T05 — bump TEE-BLK-M inventory by 50.
    "shop.T05_inventory_adjust": {
        "env": "shop", "status": "candidate",
        "plan_name": "T05_inventory_adjust", "oracle_task_id": "T05_inventory_adjust",
        "task_text": "Bump inventory of TEE-BLK-M by 50 with reason 'restock from warehouse'.",
        "steps": [
            _nav("{env_url}/admin/inventory", "Open the inventory admin"),
            _fill("SKU", "TEE-BLK-M", "Target the TEE-BLK-M variant"),
            _fill("delta", "50", "Add 50 units"),
            _fill("reason", "restock from warehouse", "Set the adjustment reason"),
            _submit("Apply", "Apply the inventory adjustment", required=False),
        ],
    },
}
