"""The repository must not reference any specific named customer.

Mantis is a public CUA repo. Source code, tests, deployment scripts, env
templates, and documentation all ship in the same git tree, and naming a
specific customer anywhere in that tree:

- discloses customer relationships the named customer may not want disclosed,
- bakes their internal product surface (their CRM URL, their tenant id,
  their sample credentials, their workflow file paths) into our public
  surface,
- pushes other prospects to copy that pattern verbatim instead of writing
  the integration that fits their own stack.

This test greps every git-tracked file for known customer tokens and
fails the build on a hit. When onboarding a new named customer whose
specifics need to live somewhere, put them in a private store (operator
secrets, a separate internal repo) — never in this tree.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# Tokens that, if they appear in any tracked file, indicate a customer-name
# leak. Stored as plain strings; matched case-insensitively. Add to this
# list when a new named customer is identified — never remove without a
# deliberate review.
CLIENT_TOKENS: tuple[str, ...] = (
    # vision_claude / staffai integration — the original leak source.
    "staffai",
    "vision_claude",
    "VisionClaude",  # camel-case variant
    "staffai-test-crm",
    # PopYachts — the third-party admin console the early lead-entry
    # workflow scraped against.
    "popyachts",
    "popsells",
    # Sample credentials that surfaced in early form-flow examples.
    "sarah.connor",
    "skynet99",
    # Internal product / tenant strings sprinkled through old examples.
    "robotcrm",
    "vision_claude_prod",
)


# Tracked-file paths to skip when scanning. The guard test itself must
# contain the tokens to enforce them. Lockfiles / binary blobs / vendored
# fixtures don't carry meaningful prose so they get skipped too.
ALLOWLIST_PATHS: frozenset[str] = frozenset({
    "tests/test_docs_client_isolation.py",
})


# Binary / non-text suffixes we don't grep into.
SKIP_SUFFIXES: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".mp4", ".webm", ".mov",
    ".pdf",
    ".woff", ".woff2", ".ttf", ".otf",
    ".onnx", ".gguf", ".bin", ".pt", ".safetensors",
    ".tar", ".gz", ".zip",
})


def _tracked_files() -> list[Path]:
    """Every git-tracked file the guard should scan."""
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    files: list[Path] = []
    for rel in out.stdout.splitlines():
        if not rel:
            continue
        if rel in ALLOWLIST_PATHS:
            continue
        path = REPO_ROOT / rel
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if not path.is_file():
            continue
        files.append(path)
    return files


@pytest.mark.parametrize("token", CLIENT_TOKENS)
def test_no_customer_token_in_tracked_files(token: str) -> None:
    """Each token in CLIENT_TOKENS must be absent from every tracked file.

    If this fails, replace the customer-named example with a neutral
    placeholder: ``crm.example.com`` / ``tenant_a`` / ``alice`` /
    ``<password>``. Never check the original customer name back in.
    """
    pattern = re.compile(re.escape(token), re.IGNORECASE)
    offenders: list[tuple[Path, int, str]] = []
    for path in _tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                offenders.append(
                    (path.relative_to(REPO_ROOT), line_no, line.strip())
                )

    assert not offenders, (
        f"Customer token {token!r} leaked into {len(offenders)} line(s):\n"
        + "\n".join(
            f"  {p}:{ln}  {snippet[:120]}" for p, ln, snippet in offenders
        )
        + "\n\nReplace each occurrence with a neutral placeholder. "
        "Customer-specific content does not belong in this public repo."
    )
