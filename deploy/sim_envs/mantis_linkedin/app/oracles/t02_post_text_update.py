"""t02_post_text_update — current user creates a feed post containing text + a #hashtag.

Pass criteria:
- A row in ``posts`` exists authored by user_00001 with at least one
  hashtag stored in the JSON ``hashtags`` column.
- The hashtag must be extracted (i.e. the row was created via the post
  route, which calls ``extract_hashtags``).
- An audit_log row with operation='post_created' references the post id.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any


def grade(conn: sqlite3.Connection, *, now: str,
          seed_val: int) -> dict[str, Any]:
    reasons: list[str] = []
    author = "user_00001"

    # Find the most recent post by demo that includes at least one hashtag.
    candidates = conn.execute(
        "SELECT id, body, hashtags, created_at FROM posts "
        "WHERE author_id = ? AND hashtags != '[]' "
        "ORDER BY id DESC LIMIT 5",
        (author,),
    ).fetchall()
    seed_post_ids = {f"post_{i:05d}" for i in range(1, 31)}

    # Skip seeded posts — only count posts the agent created (those have
    # ids > 30 because seed is fixed to 30).
    fresh = [r for r in candidates if r["id"] not in seed_post_ids]

    if not fresh:
        return {
            "passed": False,
            "score": 0.0,
            "reasons": [
                f"no agent-authored post for {author} with a hashtag found"
            ],
            "diff": {"expected": {
                "author_id": author, "hashtags": "<at-least-one>",
            }, "actual_candidates": [dict(r) for r in candidates]},
        }

    post = dict(fresh[0])
    tags = json.loads(post["hashtags"])
    if not tags:
        reasons.append("post.hashtags is empty after extraction")

    audit_row = conn.execute(
        "SELECT id, payload_json FROM audit_log "
        "WHERE operation = 'post_created' AND target_id = ? LIMIT 1",
        (post["id"],),
    ).fetchone()
    if not audit_row:
        reasons.append("no audit_log row for post_created")

    audit_payload: dict[str, Any] = {}
    if audit_row:
        try:
            audit_payload = json.loads(audit_row["payload_json"])
        except json.JSONDecodeError:
            audit_payload = {}

    if not post["body"].strip():
        reasons.append("post body is empty")

    passed = not reasons
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reasons": reasons or ["all checks passed"],
        "diff": {
            "post": {
                "id": post["id"], "body": post["body"], "hashtags": tags,
            },
            "audit_payload": audit_payload,
        },
    }
