"""Internal Anthropic API client (#406).

Lifted out of :class:`mantis_agent.extraction.extractor.ClaudeExtractor`
so both extraction code (``extract`` / ``find_all_listings`` /
``verify_gate``) and grounding code (``find_form_target`` /
``find_target_by_affordance`` / ``verify_dropdown_value``) share a
single Anthropic call site with one retry policy. No public re-exports
— callers import :class:`AnthropicToolUseClient` directly:

    from mantis_agent._anthropic.client import AnthropicToolUseClient

Leading underscore on the package name signals "internal shared
infrastructure" — same convention as Python's stdlib ``_collections``
/ ``_threading_local``. Callers from outside ``mantis_agent`` are not
expected to import from here.
"""
