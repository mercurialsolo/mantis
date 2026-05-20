"""Pre-commit lint that blocks customer-named content from being committed.

Per the standing rule (auto-memory:
``feedback_no_customer_names_in_tracked_files``): scripts, plans,
prompts, and docs MUST NOT carry customer brand names / target
domains / seed lead IDs. Use neutral phrasing (e.g. "remote-brain
integration") or move the artifact to a gitignored path (``/tmp/``,
``scripts/run_*_with_proxy.py``, ``plans/``).

This script is the pre-commit gate. Run with a list of paths
(default: ``git diff --cached --name-only --diff-filter=ACM``) and
exits non-zero on any violation.

Usage::

    # Manual run on staged files
    python scripts/lint_no_customer_info.py

    # Run on specific files
    python scripts/lint_no_customer_info.py FILE [FILE ...]

    # As pre-commit framework hook
    # (see .pre-commit-config.yaml)

The check runs in two passes:

1. **Path block** — filenames containing a banned term anywhere are
   rejected (e.g. ``scripts/run_acme_co_scraper.py``).
2. **Content block** — file body grepped (case-insensitive) for
   banned terms; first hit emits a violation pointing at the line.

Allowlist for legitimate references:

* ``tests/`` — test files can mention customer names in isolation
  (per the memory's "tests-isolated allowlist" carve-out).
* The lint script + config themselves — they contain the banned
  terms as patterns to detect.
* ``.gitignore`` — it lists banned-pattern filenames so they're
  blocked from tracking.

Add new banned terms by editing :data:`_BANNED_TERMS` below; keep
the list short + maintained — too many false positives encourages
the obvious workaround (``# noqa``).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ── Banned-term registry ─────────────────────────────────────────────────

# Customer brands, target domains, seed lead identifiers, internal
# product nicknames that should never appear in tracked files. Match
# substring (case-insensitive). Keep this list audit-able.
_BANNED_TERMS: tuple[str, ...] = (
    # Customer brand names + target sites used in adhoc scripts
    "boattrader",
    "staffai",
    "staff-ai",
    "staff_crm",
    "popyachts",
    # Customer-named test-fixture domains
    "staffai-test-crm",
    # Customer-side seed lead identifiers (these appear in adhoc
    # submit scripts and burn into git history if not gated)
    "sentinel prime",          # SEED_LEAD_BODY robot_name from staff-crm
    "pinnacle robotics",       # SEED_LEAD_BODY company
    "jordan.reyes@pinnacle",   # SEED_LEAD_BODY contact_email
)

# Paths exempt from content scanning. These files legitimately
# contain banned terms (the lint script's own pattern list, the
# gitignore that names blocked files, tests that exercise customer
# integrations in isolation).
_ALLOWED_PATHS: tuple[str, ...] = (
    "scripts/lint_no_customer_info.py",
    ".pre-commit-config.yaml",
    ".gitignore",
    "tests/",
)

# Path-block: filename substrings that are forbidden even before
# content scanning. Catches drive-by additions like
# ``scripts/run_boattrader_urlnav.py``.
_BANNED_PATH_FRAGMENTS: tuple[str, ...] = (
    "boattrader",
    "staffai",
    "staff_crm",
    "popyachts",
)


# ── Scanning ─────────────────────────────────────────────────────────────


def _is_allowed_path(path: str) -> bool:
    return any(path == p or path.startswith(p) for p in _ALLOWED_PATHS)


def _staged_files() -> list[str]:
    """Default: files staged for the next commit (added / copied / modified)."""
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _scan_file(path: str) -> list[str]:
    """Return a list of violation messages for ``path``. Empty list = clean."""
    violations: list[str] = []

    # Path-block first — cheap.
    fname_lc = path.lower()
    for frag in _BANNED_PATH_FRAGMENTS:
        if frag in fname_lc and not _is_allowed_path(path):
            violations.append(
                f"{path}: filename contains banned customer term {frag!r} "
                f"(rename to a neutral term or move to a gitignored path)"
            )

    # Skip content scan for binary / allowed paths.
    if _is_allowed_path(path):
        return violations
    p = Path(path)
    if not p.exists() or not p.is_file():
        return violations

    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return violations

    text_lc = text.lower()
    for term in _BANNED_TERMS:
        if term.lower() not in text_lc:
            continue
        # Locate first occurrence + line number for a useful error.
        for line_no, line in enumerate(text.splitlines(), 1):
            if term.lower() in line.lower():
                snippet = line.strip()[:120]
                violations.append(
                    f"{path}:{line_no}: banned customer term {term!r} "
                    f"in line {snippet!r}"
                )
                break  # one report per term per file is enough

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files", nargs="*",
        help="Paths to scan (defaults to staged files)",
    )
    args = parser.parse_args(argv)

    files = args.files or _staged_files()
    if not files:
        return 0

    all_violations: list[str] = []
    for f in files:
        all_violations.extend(_scan_file(f))

    if not all_violations:
        return 0

    print(
        "\n❌  Customer-info pre-commit gate failed "
        "(see auto-memory: feedback_no_customer_names_in_tracked_files):\n",
        file=sys.stderr,
    )
    for v in all_violations:
        print(f"  {v}", file=sys.stderr)
    print(
        "\nFix options:\n"
        "  1. Rename the file / strip the banned term + use neutral phrasing\n"
        "  2. Move the file to a gitignored path (e.g. scripts/run_*_with_proxy.py,\n"
        "     plans/, /tmp/, ~/scratch/)\n"
        "  3. If this is a legitimate test, place it under tests/ (allowlisted)\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
