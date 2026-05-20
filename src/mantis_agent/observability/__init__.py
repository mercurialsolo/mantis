"""Mantis observability adapters (#509).

Vendors out per-run telemetry that's optional at install time. Today
this is a single Augur adapter (DebugSession wrapper); future sinks
(OpenTelemetry, custom JSONL) plug in the same way.
"""
