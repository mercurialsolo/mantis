"""Backend registry — resolve ``--runtime <name>`` to a :class:`RuntimeBackend`.

Trivial map keyed on the canonical name. Splitting this out of
``__init__.py`` keeps the import cost down: the local backend pulls in
``subprocess`` + ``socket``, the modal backend lazy-imports the Modal
SDK, e2b is a stub. A direct ``from .local import LocalBackend`` in
``__init__.py`` would load all three on every import; the registry
imports only the one the caller asked for.
"""

from __future__ import annotations

from .runtime import RuntimeBackend


def get_backend(name: str) -> RuntimeBackend:
    """Return the backend registered under ``name``.

    ``ValueError`` for an unknown name — the CLI's ``--runtime`` flag
    uses ``argparse`` ``choices=`` to short-circuit invalid values, so
    this is the belt-and-braces check for direct callers.
    """
    if name == "local":
        from .local import LocalBackend

        return LocalBackend()
    if name == "modal":
        from .modal_backend import ModalBackend

        return ModalBackend()
    if name == "e2b":
        from .e2b import E2BBackend

        return E2BBackend()
    raise ValueError(
        f"unknown runtime backend: {name!r}. "
        f"Pick one of: {', '.join(list_backends())}."
    )


def list_backends() -> list[str]:
    """Names accepted by :func:`get_backend`."""
    return ["local", "modal", "e2b"]
