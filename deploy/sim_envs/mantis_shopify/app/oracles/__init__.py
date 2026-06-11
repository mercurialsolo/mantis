"""Oracles — server-side graders for mantis-shopify.

Each oracle reads DB state + audit_log. Never the agent transcript.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Callable

from . import (
    t01_submit_plus_lead,
    t02_invite_staff_member,
    t03_export_payouts_csv,
    t04_create_support_ticket,
    t05_update_business_email,
    t06_view_payout_detail,
    t07_dismiss_emergency_banner,
    t08_directory_request_review,
    t09_search_stores_filter,
    t10_submit_pos_referral,
    t11_view_store_detail,
    t12_view_catalog_detail,
    t13_view_merchant_order,
    t14_fulfill_merchant_order,
)

GraderFn = Callable[..., dict[str, Any]]

GRADERS: dict[str, GraderFn] = {
    "t01": t01_submit_plus_lead.grade,
    "t01_submit_plus_lead": t01_submit_plus_lead.grade,
    "t02": t02_invite_staff_member.grade,
    "t02_invite_staff_member": t02_invite_staff_member.grade,
    "t03": t03_export_payouts_csv.grade,
    "t03_export_payouts_csv": t03_export_payouts_csv.grade,
    "t04": t04_create_support_ticket.grade,
    "t04_create_support_ticket": t04_create_support_ticket.grade,
    "t05": t05_update_business_email.grade,
    "t05_update_business_email": t05_update_business_email.grade,
    "t06": t06_view_payout_detail.grade,
    "t06_view_payout_detail": t06_view_payout_detail.grade,
    "t07": t07_dismiss_emergency_banner.grade,
    "t07_dismiss_emergency_banner": t07_dismiss_emergency_banner.grade,
    "t08": t08_directory_request_review.grade,
    "t08_directory_request_review": t08_directory_request_review.grade,
    "t09": t09_search_stores_filter.grade,
    "t09_search_stores_filter": t09_search_stores_filter.grade,
    "t10": t10_submit_pos_referral.grade,
    "t10_submit_pos_referral": t10_submit_pos_referral.grade,
    "t11": t11_view_store_detail.grade,
    "t11_view_store_detail": t11_view_store_detail.grade,
    "t12": t12_view_catalog_detail.grade,
    "t12_view_catalog_detail": t12_view_catalog_detail.grade,
    "t13": t13_view_merchant_order.grade,
    "t13_view_merchant_order": t13_view_merchant_order.grade,
    "t14": t14_fulfill_merchant_order.grade,
    "t14_fulfill_merchant_order": t14_fulfill_merchant_order.grade,
}


def grade(task_id: str, conn: sqlite3.Connection, *, now: str,
          seed_val: int) -> dict[str, Any]:
    fn = GRADERS.get(task_id)
    if fn is None:
        return {
            "passed": False,
            "score": 0.0,
            "task_id": task_id,
            "reasons": [f"no oracle registered for task_id={task_id!r}"],
            "diff": {},
        }
    result = fn(conn, now=now, seed_val=seed_val)
    result.setdefault("task_id", task_id)
    return result
