"""Externalized prompts for the Mantis agent.

Prompt bodies live as ``.txt`` files alongside this module in
``files/<name>.txt``. Keeping the bodies as plain text makes them easy
to diff in code review, A/B-test wording without forking a brain
module, and edit without touching Python escaping rules.

Adding a new prompt:
    1. Drop the body in ``files/<name>.txt``
    2. Add ``<name>`` to :data:`_PROMPT_NAMES`
    3. (Optional) register placeholder defaults in
       :data:`_PROMPT_PLACEHOLDER_DEFAULTS` so unset tokens collapse to
       sensible values rather than leaking ``__TOKEN__`` into the model
       prompt
    4. Reference it from the call site via :func:`load_prompt`

Substitution uses double-underscore placeholders like ``__SCREEN_WIDTH__``
so the prompt body can freely contain ``{`` and ``}`` (e.g. JSON examples)
without escaping.

Operator override
-----------------
Set the env var ``MANTIS_PROMPTS_DIR=/path/to/prompts`` to swap an
individual prompt without forking the package. The loader looks up
``<dir>/<name>.txt`` first; if found, the file content overrides the
in-tree template. This lets a tenant tune wording (entity name, locale)
without redeploying the wheel.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

_FILES_DIR = Path(__file__).parent / "files"

# Names of every prompt template shipped in ``files/``. Order is irrelevant;
# tests assert membership, not sequence.
_PROMPT_NAMES: tuple[str, ...] = (
    "system_v1",
    "gemma4_system",
    "holo3_system",
    "fara_system",
    "claude_system",
    "opencua_system",
    "llamacpp_system",
    # Action-side prompts (issue #224 + agentic-recovery follow-ups).
    # Externalised so plan authors / operators can A/B-test the wording
    # without forking Python modules. ``MANTIS_PROMPTS_DIR`` overrides
    # apply to these the same way as the system templates.
    "recovery_analysis",
    "derive_objective",
    "find_form_target",
)

# Placeholder defaults applied before caller-supplied substitutions. Keeps
# domain-neutral templates clean of unresolved ``__TOKEN__`` markers when the
# caller doesn't provide a value (e.g. ``__EXAMPLES_BLOCK__`` collapses to
# empty string by default; a domain harness can pass a populated block).
_PROMPT_PLACEHOLDER_DEFAULTS: dict[str, dict[str, str]] = {
    "holo3_system": {"examples_block": ""},
}


def _read_packaged_template(name: str) -> str:
    """Read ``files/<name>.txt`` shipped alongside this module."""
    return (_FILES_DIR / f"{name}.txt").read_text(encoding="utf-8")


# In-tree templates loaded from disk at module import. Operators who tweak a
# ``files/<name>.txt`` see the change after the next process start; no edits
# to this Python module required.
_PROMPTS: dict[str, str] = {name: _read_packaged_template(name) for name in _PROMPT_NAMES}


def _override_dir() -> Path | None:
    """Resolve ``MANTIS_PROMPTS_DIR`` to a Path if set and existing."""
    raw = os.environ.get("MANTIS_PROMPTS_DIR", "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.is_dir() else None


def _read_template(name: str) -> str | None:
    """Resolve the raw template text, honouring the operator override.

    Returns ``None`` when the name is unknown and no override file exists.
    The text is the *raw* template — placeholders are not substituted here.
    """
    override = _override_dir()
    if override is not None:
        candidate = override / f"{name}.txt"
        if candidate.is_file():
            try:
                return candidate.read_text(encoding="utf-8")
            except OSError:
                pass
    return _PROMPTS.get(name)


def load_prompt(name: str, **substitutions: object) -> str:
    """Load a prompt by name and substitute ``__KEY__`` placeholders.

    Resolution order:

    1. ``MANTIS_PROMPTS_DIR/<name>.txt`` if the env var is set and the
       file exists. Lets operators override individual prompts without
       forking the wheel.
    2. The in-tree template at ``files/<name>.txt``.

    For each prompt, :data:`_PROMPT_PLACEHOLDER_DEFAULTS` is applied
    first, then caller-supplied ``substitutions`` override. Substitution
    keys are normalised to uppercase. Values are ``str()``-coerced.

    Example::

        load_prompt("system_v1", screen_width=1280, screen_height=720, password="hunter2")

    Raises ``KeyError`` if the name is unknown and no override file exists.
    """
    text = _read_template(name)
    if text is None:
        raise KeyError(
            f"Unknown prompt: {name!r}. Available: {sorted(_PROMPTS)}; "
            f"override dir: {_override_dir() or '(unset)'}"
        )

    merged: dict[str, str] = dict(_PROMPT_PLACEHOLDER_DEFAULTS.get(name, {}))
    for key, value in substitutions.items():
        merged[key.lower()] = str(value)
    for key, value in merged.items():
        text = text.replace(f"__{key.upper()}__", value)
    return text.strip()


def list_prompts() -> list[str]:
    """Names of all in-tree prompts, sorted. Override files are not enumerated."""
    return sorted(_PROMPTS)


def prompt_version(name: str) -> str:
    """Return an 8-char SHA1 of the prompt content as currently resolved (#127).

    Honors ``MANTIS_PROMPTS_DIR`` overrides — operators who replace a prompt
    should see a different SHA so prompt regressions are attributable. The
    raw template is hashed (no substitution applied), so the version tracks
    the source-of-truth content rather than per-run rendering.

    Returns ``"unknown"`` for names not registered and not present in the
    override dir, so this is safe to call eagerly at run start without
    crashing on a typo.
    """
    text = _read_template(name)
    if text is None:
        return "unknown"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def current_prompt_versions() -> dict[str, str]:
    """``{prompt_name: short_sha}`` for every in-tree prompt (#127).

    Intended for run-start telemetry. Override files for unknown names are
    *not* enumerated — only registered prompts contribute, matching
    :func:`list_prompts` semantics.
    """
    return {name: prompt_version(name) for name in _PROMPTS}


# Module-level constants — rendered with placeholder defaults applied. Tests
# and direct importers (``from mantis_agent.prompts import HOLO3_SYSTEM``)
# rely on these. The .txt file is the canonical source; these are derived.
SYSTEM_V1 = load_prompt("system_v1")
GEMMA4_SYSTEM = load_prompt("gemma4_system")
HOLO3_SYSTEM = load_prompt("holo3_system")
FARA_SYSTEM = load_prompt("fara_system")
CLAUDE_SYSTEM = load_prompt("claude_system")
OPENCUA_SYSTEM = load_prompt("opencua_system")
LLAMACPP_SYSTEM = load_prompt("llamacpp_system")


__all__ = [
    "load_prompt",
    "list_prompts",
    "prompt_version",
    "current_prompt_versions",
    "SYSTEM_V1",
    "GEMMA4_SYSTEM",
    "HOLO3_SYSTEM",
    "FARA_SYSTEM",
    "CLAUDE_SYSTEM",
    "OPENCUA_SYSTEM",
    "LLAMACPP_SYSTEM",
]
