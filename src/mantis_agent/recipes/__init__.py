"""Mantis recipes — vertical-specific bundles (schema + plan + rewards).

A recipe is a self-contained directory under ``mantis_agent.recipes`` that
provides:

- ``schema.py`` exposing ``SCHEMA: ExtractionSchema``
- ``plan.json`` — a valid micro-plan
- optionally ``site_config.py`` (``SITE_CONFIG: SiteConfig``),
  ``rewards.py`` (``REWARD: RewardFn``), and ``verifier.py``

The core never imports from a specific recipe. Plans declare a recipe by
name; the loader resolves it via ``importlib`` at run time.

Per issue #224 the recipe is shifting from "primary configuration source"
to "production-hardening overlay" on top of derived ``ExtractionSchema``
and probe-derived ``SiteConfig``. ``load_schema`` and
``load_site_config`` are the lookup half; the overlay half lives on the
respective dataclasses (see ``ExtractionSchema.overlay`` /
``SiteConfig.overlay``).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..extraction import ExtractionSchema
    from ..site_config import SiteConfig


def load_schema(name: str, *, tenant_id: str | None = None) -> "ExtractionSchema":
    """Resolve a recipe schema by name.

    Lookup precedence (#809):

    1. If ``tenant_id`` is supplied, check the tenant's runtime recipe
       directory at ``/data/tenants/<tenant>/recipes/<name>.json``
       first. Runtime recipes registered via ``POST /v1/recipes``
       can override a code-shipped recipe of the same name without
       affecting other tenants.
    2. Fall back to ``mantis_agent.recipes.<name>.schema.SCHEMA`` for
       the code-shipped recipe.

    Raises ``ModuleNotFoundError`` if neither path resolves.
    """
    if tenant_id:
        # Lazy import to keep the static-recipe lookup path independent
        # of the runtime store's filesystem assumptions.
        from . import runtime_store

        runtime = runtime_store.load_schema(tenant_id, name)
        if runtime is not None:
            return runtime
    mod = importlib.import_module(f"{__name__}.{name}.schema")
    try:
        return mod.SCHEMA
    except AttributeError as exc:
        raise ModuleNotFoundError(
            f"recipe {name!r} does not export SCHEMA in schema.py"
        ) from exc


def load_site_config(name: str) -> "SiteConfig | None":
    """Resolve ``mantis_agent.recipes.<name>.site_config.SITE_CONFIG``.

    Symmetric with :func:`load_schema`. ``site_config.py`` is optional —
    a recipe may ship a ``SCHEMA`` without a ``SITE_CONFIG`` (e.g. when
    the URL / pagination shape is generic enough for the probe-derived
    default to suffice). Returns ``None`` in that case rather than
    raising, so callers can do::

        site = SiteConfig.from_probe(probe).overlay(
            recipes.load_site_config(name)
        )

    and have ``overlay(None)`` behave as a no-op.

    Raises ``ModuleNotFoundError`` only when the recipe directory itself
    doesn't exist — distinguishing "recipe absent" from "recipe present,
    no site config".
    """
    try:
        mod = importlib.import_module(f"{__name__}.{name}.site_config")
    except ModuleNotFoundError as exc:
        # Distinguish "recipe directory missing" (re-raise) from
        # "site_config.py missing" (return None). importlib raises the
        # same exception type for both; we look at the args.
        missing = getattr(exc, "name", "") or ""
        if missing.endswith(f".{name}"):
            raise
        if missing == f"{__name__}.{name}.site_config":
            return None
        # Some other intermediate import error — surface it.
        raise
    return getattr(mod, "SITE_CONFIG", None)


def load_loop_overrides(name: str) -> dict[str, object]:
    """Resolve ``mantis_agent.recipes.<name>.loop_overrides.LOOP_OVERRIDES``.

    Same shape contract as :func:`load_site_config`: missing
    ``loop_overrides.py`` returns an empty dict (recipe ships no loop
    customizations), missing recipe directory raises. The empty-dict
    path lets the caller spread the result into a ``LoopConfig`` without
    branching on whether the recipe ships loop overrides::

        overrides = recipes.load_loop_overrides(recipe_name)
        loop_cfg = LoopConfig(iteration_intent=..., **overrides)

    Keys must match ``gym.workflow_runner.LoopConfig`` field names. See
    :mod:`mantis_agent.recipes.marketplace_listings.loop_overrides` for
    the reference shape (issue #463).
    """
    try:
        mod = importlib.import_module(f"{__name__}.{name}.loop_overrides")
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "") or ""
        if missing.endswith(f".{name}"):
            raise
        if missing == f"{__name__}.{name}.loop_overrides":
            return {}
        raise
    overrides = getattr(mod, "LOOP_OVERRIDES", None)
    if overrides is None:
        return {}
    return dict(overrides)
