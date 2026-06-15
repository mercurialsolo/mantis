"""Serving-side helpers for the Mantis CUA model server.

Currently houses :mod:`mantis_agent.serving.lora_serving` — the logic that lets
the ``/v1/predict`` endpoint serve a **base + trained LoRA adapter** so the
promotion gate (#894/#911) can evaluate a challenger checkpoint against the
champion. The pure-logic surface lives here (unit-testable, no GPU); the Modal
GPU glue in ``deploy/modal/modal_cua_server.py`` calls into it.
"""
