"""Public docs must not reference any specific named client.

Mantis hosts multiple integrations. Public documentation has to work for
any of them without disclosing who else is on the platform — naming a
specific client (their CRM URL, their tenant id, their internal file
paths, their sample credentials) in the public site:

- discloses customer relationships the named client may not want disclosed,
- bakes their internal product surface into our public surface,
- pushes other clients to copy that pattern verbatim instead of writing
  the integration that fits their own stack.

Generic shape goes in ``docs/``. Client-named specifics go in
``internal-docs/``. This test enforces the boundary by grepping
``docs/`` for known client tokens.

When onboarding a new named client whose docs live in
``internal-docs/``, add their token strings to ``CLIENT_TOKENS`` below.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCS_DIR = REPO_ROOT / "docs"

# Case-insensitive substrings that, if they appear anywhere in
# ``docs/**.md``, indicate a client-name leak.
#
# Add to this list when onboarding a new named client. Removing a token
# is a deliberate decision: we believe the name has become a generic
# industry term (rare).
CLIENT_TOKENS: tuple[str, ...] = (
    # vision_claude / staffai integration — the canonical leak source.
    "staffai",
    "vision_claude",
    "VisionClaude",  # camel-case variants
    "staffai-test-crm",
    # Sample creds the staffai team used in early plan examples.
    "sarah.connor",
    "skynet99",
    # Internal product names previously sprinkled through examples.
    "robotcrm",
    # Tenant IDs surfaced from production configs.
    "vision_claude_prod",
)


def _public_docs() -> list[Path]:
    return sorted(PUBLIC_DOCS_DIR.rglob("*.md"))


@pytest.mark.parametrize("token", CLIENT_TOKENS)
def test_public_docs_do_not_leak_client_token(token: str) -> None:
    """Each token in CLIENT_TOKENS must be absent from every public doc.

    If this fails, the offending file should either:
      a) be relocated under ``internal-docs/`` (whole-file leak), OR
      b) have the client-named example replaced with a neutral placeholder
         like ``crm.example.com`` / ``tenant_a`` / ``alice``.
    """
    pattern = re.compile(re.escape(token), re.IGNORECASE)
    offenders: list[tuple[Path, int, str]] = []
    for path in _public_docs():
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                offenders.append((path.relative_to(REPO_ROOT), line_no, line.strip()))

    assert not offenders, (
        f"Client token {token!r} leaked into {len(offenders)} public-doc line(s):\n"
        + "\n".join(f"  {p}:{ln}  {snippet[:120]}" for p, ln, snippet in offenders)
        + "\n\nMove the offending content to internal-docs/ or replace the "
        "client-named example with a neutral placeholder. See "
        "internal-docs/README.md for the policy."
    )


def test_internal_docs_directory_exists_and_is_excluded_from_mkdocs():
    """Sanity-check that the directory holding client-named docs exists
    and that the mkdocs config doesn't accidentally pull it in. If the
    directory layout changes, this test wants to know."""
    internal = REPO_ROOT / "internal-docs"
    assert internal.is_dir(), "internal-docs/ should exist (see internal-docs/README.md)"
    assert (internal / "README.md").is_file(), "internal-docs/README.md is the policy doc"

    mkdocs_yml = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    # mkdocs builds from docs_dir: docs — must NOT include internal-docs.
    assert "docs_dir: docs" in mkdocs_yml
    assert "internal-docs" not in mkdocs_yml, (
        "internal-docs/ must NEVER be referenced from mkdocs.yml — that "
        "would make client-named content publicly searchable."
    )
