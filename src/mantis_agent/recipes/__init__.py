"""Mantis recipes — vertical-specific bundles (schema + plan + rewards).

A recipe is a self-contained directory under ``mantis_agent.recipes`` that
provides:

- ``schema.py`` exposing ``SCHEMA: ExtractionSchema``
- ``plan.json`` — a valid micro-plan
- optionally ``rewards.py`` (``REWARD: RewardFn``) and ``verifier.py``

The core never imports from a specific recipe. Plans declare a recipe by
name; the loader resolves it via ``importlib`` at run time.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..extraction import ExtractionSchema


def load_schema(name: str) -> "ExtractionSchema":
    """Resolve ``mantis_agent.recipes.<name>.schema.SCHEMA``.

    Raises ``ModuleNotFoundError`` if the recipe doesn't exist or doesn't
    export a ``SCHEMA`` symbol.
    """
    mod = importlib.import_module(f"{__name__}.{name}.schema")
    try:
        return mod.SCHEMA
    except AttributeError as exc:
        raise ModuleNotFoundError(
            f"recipe {name!r} does not export SCHEMA in schema.py"
        ) from exc
