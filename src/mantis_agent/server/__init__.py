"""Shared HTTP-server utilities used by both Baseten and Modal deployments.

The Baseten deployment (``baseten_server/``) is the canonical
implementation; the Modal deployment (``deploy/modal/modal_cua_server.py``,
ASGI endpoint added in #342) reuses the framework-agnostic pieces here
so the two surfaces don't drift.
"""
