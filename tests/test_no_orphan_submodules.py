"""Guard against orphan submodule entries in the git index.

A gitlink (mode 160000) without a matching entry in ``.gitmodules`` is
silently destructive for downstream installers: ``pip install git+...``
and ``uv add git+...`` both run ``git submodule update --init --recursive``
unconditionally on git+ URLs, and bail with

    fatal: no submodule mapping found in .gitmodules for path '<path>'

This bug previously bit the repo when an unused benchmarking-config
gitlink stuck around without a ``.gitmodules`` mapping — every
downstream consumer of ``pip install git+https://.../mantis@<sha>``
had to work around it by cloning ``--no-recurse-submodules`` and then
``pip install /local/path``, which is fragile and surprising.

The guard:
  - Greps the index for mode-160000 entries.
  - Reads ``.gitmodules`` (if it exists) to find which paths are mapped.
  - Fails if any gitlink lacks a mapping.

If you legitimately want a submodule, add it via ``git submodule add
<url> <path>`` so ``.gitmodules`` is created and populated. If you don't
want a submodule, ``git rm --cached <path>`` removes the orphan entry.
"""

from __future__ import annotations

import configparser
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _gitlinks() -> list[str]:
    """Paths in the git index registered as submodule gitlinks (mode 160000)."""
    out = subprocess.run(
        ["git", "ls-files", "--stage"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    paths: list[str] = []
    for line in out.stdout.splitlines():
        # Format: <mode> <sha> <stage>\t<path>
        if not line.startswith("160000"):
            continue
        try:
            _, rest = line.split("\t", 1)
        except ValueError:
            continue
        paths.append(rest.strip())
    return paths


def _mapped_submodule_paths() -> set[str]:
    """Paths declared in ``.gitmodules``, if the file exists."""
    gitmodules = REPO_ROOT / ".gitmodules"
    if not gitmodules.is_file():
        return set()
    parser = configparser.ConfigParser()
    parser.read(gitmodules)
    return {
        parser.get(section, "path")
        for section in parser.sections()
        if parser.has_option(section, "path")
    }


def test_no_orphan_submodule_entries():
    """Every gitlink in the index must have a matching ``.gitmodules`` entry.

    An orphan gitlink breaks ``pip install git+...`` and ``uv add git+...``
    for every downstream caller. Catch it on the first PR that introduces it.
    """
    gitlinks = _gitlinks()
    mapped = _mapped_submodule_paths()
    orphans = [path for path in gitlinks if path not in mapped]
    assert not orphans, (
        f"Found {len(orphans)} orphan submodule gitlink(s): {orphans!r}\n\n"
        "Each gitlink (mode 160000) needs a .gitmodules entry, or pip / uv "
        "installs from this repo will fail with "
        "'fatal: no submodule mapping found in .gitmodules for path ...'.\n\n"
        "Fix one of two ways:\n"
        "  1. Remove the orphan: `git rm --cached <path>`\n"
        "  2. Convert to a real submodule: `git submodule add <url> <path>`"
    )
