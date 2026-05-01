"""Modal runtime — image, llama-server lifecycle, learnings, OSWorld harness.

Public surface (re-exported here for backwards compatibility — the
single-file ``mantis_agent.modal_runtime`` was split into a package in
PR #106 to reduce single-file size):

- :data:`image`, :data:`vol`, :data:`GEMMA4_MODEL`, :data:`GGUF_CONFIGS`
  from :mod:`.image`
- :func:`download_model`, :func:`start_llama_server` from :mod:`.llama`
- :func:`load_learnings`, :func:`get_prior_learning` from :mod:`.learnings`
- :func:`run_osworld_impl`, :func:`extract_setup_paths`,
  :func:`extract_setup_cwd`, :func:`derive_hint` from :mod:`.osworld`

Existing callers (``deploy/modal/*.py``, ``benchmarks/osworld_*.py``)
that do ``from mantis_agent.modal_runtime import image, run_osworld_impl,
vol`` keep working unchanged.
"""

from __future__ import annotations

from .image import GEMMA4_MODEL, GGUF_CONFIGS, image, vol
from .learnings import get_prior_learning, load_learnings
from .llama import download_model, start_llama_server
from .osworld import (
    derive_hint,
    extract_setup_cwd,
    extract_setup_paths,
    run_osworld_impl,
)

__all__ = [
    # Image / volume / model selection
    "image",
    "vol",
    "GEMMA4_MODEL",
    "GGUF_CONFIGS",
    # llama-server lifecycle
    "download_model",
    "start_llama_server",
    # Learnings persistence
    "load_learnings",
    "get_prior_learning",
    # OSWorld harness
    "run_osworld_impl",
    "extract_setup_paths",
    "extract_setup_cwd",
    "derive_hint",
]
