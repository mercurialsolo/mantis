"""Claude-based data extraction — read structured data from screenshots.

Uses Claude Sonnet to extract listing data from a single screenshot.
Same API pattern as ClaudeGrounding but for data extraction instead of
click targeting.

Architecture::

    Holo3 navigates → clicks listing → screenshot captured
      ↓
    ClaudeExtractor.extract(screenshot) → structured data
      ↓
    Holo3 navigates back

Cost: ~$0.003-0.005 per extraction call (1 screenshot + short prompt).
Called once or twice per listing (top of page + after scrolling).

Usage::

    extractor = ClaudeExtractor()
    data = extractor.extract(screenshot)

This module owns ``ClaudeExtractor`` and the prompt constants. The
schema, result, and spam helpers were split out under :mod:`.schema`,
:mod:`.result`, and :mod:`.spam` in PR #105.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, ClassVar

from PIL import Image

from .._anthropic.client import (
    _TRANSIENT_STATUS_CODES,
    AnthropicToolUseClient,
    _credit_claude_time,
    _retry_delay,
    credit_claude_tokens_from_response,
    encode_screenshot_for_claude,
)
from ..prompts import load_prompt as _load_prompt
from .result import ExtractionResult
from .schema import ExtractionSchema
from .spam import contains_dealer_text, parse_bool, seller_looks_like_dealer

# Default model for cheap binary verifier calls (verify_gate, StepVerifier).
# Haiku 4.5 is roughly 10× cheaper per token than Opus 4.7 and well-
# capable of "does the screenshot show X?" boolean judgements. Operators
# can pin a stronger model per-deployment via ``MANTIS_VERIFY_MODEL``;
# kept as an env override so a regression can be rolled back without a
# code change. See #421.
_VERIFY_MODEL = os.environ.get("MANTIS_VERIFY_MODEL", "claude-haiku-4-5-20251001")

# Escalation model: re-asked once on Haiku ``passed=False`` to avoid
# false-negative recovery loops (#421 §3). Recovery is the real cost
# spike (re-navigate + re-extract + re-verify ≈ $0.50+) so a ~$0.003
# escalation pays for itself many times over.
_VERIFY_ESCALATION_MODEL = os.environ.get(
    # Was ``claude-opus-4-7`` until 2026-05-30 — observed cost was 73 %
    # of total Claude spend across the 3-round BoatTrader sweep (Opus
    # 4.7 escalations: 63 calls × ~$0.047 = $2.97 of $4.07). Sonnet
    # 4.6 is 5× cheaper per token, similar quality for gate
    # disagreement re-checks. Override via env if you need Opus back
    # for a specific run.
    "MANTIS_VERIFY_ESCALATION_MODEL", "claude-sonnet-4-6",
)

logger = logging.getLogger(__name__)


# Re-exports for backward compatibility — tests import these names from
# ``mantis_agent.extraction.extractor``. The real definitions moved to
# :mod:`mantis_agent._anthropic.client` under #406 so the same retry
# policy backs both extraction and grounding callers.
__all__ = [
    "ClaudeExtractor",
    "_TRANSIENT_STATUS_CODES",
    "_retry_delay",
    "_credit_claude_time",
]


def _coerce_coord(value: Any) -> int | None:
    """Best-effort int coercion for click coordinates returned by Claude.

    Tool_use ``input_schema`` requires ``"type": "integer"`` for
    coordinate fields, but the model occasionally emits values as
    strings with stray whitespace / trailing commas (canonical
    failure: ``"x": "296, "`` observed on long-prompt retries that
    fed failure-history into the search). Crashing the run on
    those cases — instead of treating them as ``not_found`` — was
    a sharp edge surfaced by the priority-field staff-crm rerun.

    Returns the parsed int, or ``None`` when the value can't be
    coerced. Caller treats ``None`` as the same not-found path as
    a zero-coordinate response.
    """
    if isinstance(value, bool):
        # bool is a subclass of int — explicit reject so True/False
        # don't smuggle in as 1 / 0.
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        # Strip whitespace and trailing punctuation (commas / semicolons
        # / closing brackets) the model occasionally appends.
        cleaned = value.strip().rstrip(",;]}").strip()
        if not cleaned:
            return None
        try:
            return int(cleaned)
        except ValueError:
            try:
                return int(float(cleaned))
            except ValueError:
                return None
    return None

# Generic fallback prompts used when ClaudeExtractor is constructed without
# a schema. They describe the extractor's job in entity-neutral language and
# rely on the caller's plan/intent to provide context. Application-specific
# behaviour (boat listings, job postings, real-estate) MUST come through an
# explicit ExtractionSchema — the prompts below contain no hardcoded labels,
# field names, or industry verbs.
#
# Prompt bodies live under ``mantis_agent.prompts.files/*.txt`` so wording
# tweaks (Haiku-tuned few-shot, locale variants) get a SHA bump via
# :func:`mantis_agent.prompts.prompt_version` and an A/B-able override via
# the ``MANTIS_PROMPTS_DIR`` env var. These module-level constants are kept
# for back-compat (``from mantis_agent.extraction import EXTRACT_PROMPT``)
# and resolve to the same text the loader returns.

EXTRACT_PROMPT = _load_prompt("extract_listing")
EXTRACT_SCROLLED_PROMPT = _load_prompt("extract_listing_scrolled")
EXTRACT_MULTI_SCREENSHOT_PROMPT = _load_prompt("extract_listing_multi")
FIND_LISTING_CONTENT_CONTROL_PROMPT = _load_prompt("find_listing_content_control")


class ClaudeExtractor:
    """Extract structured data from listing screenshots using Claude Sonnet.

    Same API pattern as ClaudeGrounding — cheap, fast, vision-capable.
    Called 1-2 times per listing (top screenshot + scrolled screenshot).

    When ``schema`` is provided, prompts are generated dynamically from the
    schema fields instead of using the hardcoded BoatTrader constants.
    Callers that construct ``ClaudeExtractor()`` with no args get the existing
    BoatTrader behavior unchanged.
    """

    def __init__(
        self,
        api_key: str = "",
        # Default extraction model: Sonnet 4.6. Switched from Opus 4.7
        # after observing extraction failures on populated detail pages
        # (boattrader run 20260522_204618_706eca3d returned 0 leads
        # despite year/make/model/price all visible in the screenshot).
        # Sonnet is the right level for structured-field tool_use
        # extraction — fast, cheap, and proven on similar shapes.
        # Verifier escalation path (haiku → opus on disagreement) is
        # unchanged.
        model: str = "claude-sonnet-4-6",
        schema: ExtractionSchema | None = None,
        form_target_model: str = "",
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        # #434 follow-up: the initial Haiku default was reverted after a
        # Modal staff-crm comparison showed Haiku-grounding produced
        # MORE total cost than Opus-grounding because lower-accuracy
        # grounding triggers more retries (75→93 grounding calls
        # observed on otherwise-identical halt runs). The kwarg-driven
        # split mechanism stays so operators can A/B-test or override
        # per-deployment; the default now matches the extractor's main
        # model so existing callers see no behaviour change. See #434
        # for the data + decision.
        self.form_target_model = form_target_model or model
        self.schema = schema
        self.debug_dir = os.environ.get("MANTIS_DEBUG_DIR", "/data/screenshots/claude_debug")
        # Extraction / dropdown-verify / find_listing_content client —
        # uses the heavier ``model`` for tasks where misextraction is
        # expensive.
        self._client = AnthropicToolUseClient(
            api_key=self.api_key,
            model=self.model,
            log_prefix="ClaudeExtractor",
        )
        # Verifier clients (#421). ``verify_gate`` is binary yes/no —
        # well within Haiku 4.5's range and ~85% cheaper per call. The
        # escalation client re-asks Opus once on a Haiku FAIL so we
        # don't trigger expensive recovery loops on a Haiku false-
        # negative. Both clients share the extractor's api_key and
        # tag into a distinct time bucket so cost reports can show
        # ``claude_verify_haiku`` vs ``claude_verify_opus_escalation``.
        self._verify_client = AnthropicToolUseClient(
            api_key=self.api_key,
            model=_VERIFY_MODEL,
            log_prefix="ClaudeVerifyHaiku",
        )
        self._verify_escalation_client = AnthropicToolUseClient(
            api_key=self.api_key,
            model=_VERIFY_ESCALATION_MODEL,
            log_prefix="ClaudeVerifyOpus",
        )
        # Phone-only re-extract client (Sonnet). When the primary Haiku
        # extract returns ``phone=""`` on a listing that otherwise
        # extracted cleanly (year/make/model populated), this client
        # runs a focused single-field "find the phone in the description"
        # pass against the same multi-screenshot bundle. Empirically
        # Haiku misses phones embedded in description body text at a
        # ~83% rate (Chrome MCP audit 2026-06-03 on 22 sampled URLs);
        # Sonnet is much better at reading small body-text characters
        # in screenshots. See experiments/phone_extract_fix/PLAN.md.
        self._phone_reextract_client = AnthropicToolUseClient(
            api_key=self.api_key,
            model=os.environ.get("MANTIS_PHONE_REEXTRACT_MODEL", _VERIFY_ESCALATION_MODEL),
            log_prefix="ClaudePhoneReextract",
        )

    # ── Dynamic prompt generation from schema ─────────────────────

    def _get_extract_prompt(self) -> str:
        """Return extraction prompt — dynamic from schema or legacy hardcoded."""
        if not self.schema:
            return EXTRACT_PROMPT
        s = self.schema
        return (
            f"Look at this screenshot of a {s.entity_name} page.\n\n"
            f"Extract ALL of the following data visible on the page:\n\n"
            f"{s.field_descriptions()}\n\n"
            f"For phone numbers: look in description, contact, or seller sections.\n"
            f"If no phone is visible, return phone as \"\".\n\n"
            f"Output ONLY valid JSON:\n{s.json_template()}"
        )

    def _get_multi_extract_prompt(self) -> str:
        """Return multi-screenshot extraction prompt."""
        if not self.schema:
            return EXTRACT_MULTI_SCREENSHOT_PROMPT
        s = self.schema
        spam_rules = ", ".join(f'"{ind}"' for ind in s.spam_indicators[:6])
        return (
            f"You are looking at multiple screenshots from the SAME {s.entity_name} page.\n"
            f"They were captured at different scroll positions.\n\n"
            f"Extract ALL fields visible across ALL screenshots:\n\n"
            f"{s.field_descriptions()}\n\n"
            f"Phone search priority:\n"
            f"- Description text, contact sections, detail areas\n"
            f"- Phone reveal buttons if visible\n"
            f"- International numbers are valid\n\n"
            f"{s.spam_label.title()} detection:\n"
            f"- is_spam=true for {s.spam_label} listings containing: {spam_rules}\n\n"
            f"If no phone is visible, use \"\".\n\n"
            f"Output ONLY valid JSON:\n{s.json_template()}"
        )

    def _get_find_listings_prompt(self, skip_titles: list[str] | None = None) -> str:
        """Return find-all-listings prompt.

        #584: when the schema declares ``listing_card_exclusions``, enumerate
        them explicitly as an EXCLUDE block so Claude vision doesn't return
        marketing CTAs (financing prompts, insurance quotes, sponsored
        boats, etc.) as "listings". Without this, a click step picks
        coords on a CTA card → navigates to a marketing page → wrong-page
        extract → halt_reason cycle.
        """
        if not self.schema:
            return ""  # Legacy path uses hardcoded prompt inline
        s = self.schema
        skip = ""
        if skip_titles:
            skip = "\n\nSKIP these already-processed items:\n" + "\n".join(f"- {t}" for t in skip_titles[:20])
        exclusions_block = ""
        if s.listing_card_exclusions:
            exclusions_block = (
                "\n\nEXCLUDE these specifically (they look like listings "
                "but aren't):\n"
                + "\n".join(f"- {e}" for e in s.listing_card_exclusions)
            )
        return (
            f"Look at this screenshot of a search results page.\n\n"
            f"Find ALL visible {s.entity_name} cards/items on this page.\n"
            f"For each, report the center coordinates and title text.\n\n"
            f"SKIP: sponsored, advertisement, {s.spam_label} inventory."
            f"{exclusions_block}\n"
            f"ONLY include organic {s.entity_name} results that show a "
            f"product identity (year/make/title) AND a price."
            f"{skip}\n\n"
            f"Output ONLY valid JSON:\n"
            f"{{\"listings\": [[x, y, \"title text\"], ...], \"pagination_y\": null_or_number}}"
        )

    def _get_content_control_prompt(self) -> str:
        """Return find-content-control prompt."""
        if not self.schema:
            return FIND_LISTING_CONTENT_CONTROL_PROMPT
        s = self.schema
        allowed = ", ".join(f'"{c}"' for c in s.allowed_controls)
        forbidden = ", ".join(f'"{c}"' for c in s.forbidden_controls)
        return (
            f"Look at this {s.entity_name} page screenshot.\n\n"
            f"Find ONE visible control that should be clicked to reveal more\n"
            f"seller-supplied text or contact information.\n\n"
            f"Prefer these safe targets:\n- {allowed}\n\n"
            f"Do NOT choose: {forbidden}\n\n"
            f"OUTPUT FORMAT — strict JSON, no prose preamble, no commentary,\n"
            f"no markdown fence. The shape is exactly five string-keyed\n"
            f"fields. ``x`` and ``y`` are separate integer keys — never a\n"
            f"tuple, never two unlabeled positional values like\n"
            f"``\"x\": 302, 43``.\n\n"
            f"If a safe control IS visible — concrete example:\n"
            f"{{\"x\": 740, \"y\": 320, \"action\": \"expand_description\", "
            f"\"label\": \"Show more\", \"reason\": \"see-more chevron in description\"}}\n\n"
            f"If NO safe control is visible — exact response:\n"
            f"{{\"x\": 0, \"y\": 0, \"action\": \"none\", \"label\": \"\", \"reason\": \"none visible\"}}\n\n"
            f"Substitute the integers, label, and reason with what you "
            f"actually see — keep every key spelled exactly as shown above. "
            f"Allowed values for ``action``: expand_description, show_phone, "
            f"none."
        )

    def _parse_schema_result(self, data: dict[str, Any]) -> ExtractionResult:
        """Parse Claude response dict into ExtractionResult with schema fields."""
        # Populate both named fields (backward compat) and generic dict
        result = ExtractionResult(
            year=str(data.get("year", "")),
            make=str(data.get("make", "")),
            model=str(data.get("model", "")),
            price=str(data.get("price", "")),
            phone=str(data.get("phone", "")),
            url=str(data.get("url", "")),
            seller=str(data.get("seller", "")),
            is_dealer=parse_bool(data.get("is_dealer") or data.get("is_spam", False)),
            raw_response=json.dumps(data),
            _schema=self.schema,
        )
        # Fill generic fields from schema
        if self.schema:
            for f in self.schema.fields:
                name = f["name"]
                result.extracted_fields[name] = str(data.get(name, ""))
        return result

    def _debug_path(self, stem: str, suffix: str) -> str:
        """Build a writable debug artifact path."""
        candidates = [self.debug_dir, "/tmp/mantis_debug"]
        for base_dir in candidates:
            try:
                os.makedirs(base_dir, exist_ok=True)
                return os.path.join(base_dir, f"{stem}_{int(time.time())}{suffix}")
            except OSError:
                continue
        return os.path.join("/tmp", f"{stem}_{int(time.time())}{suffix}")

    def _call(
        self,
        screenshot: Image.Image,
        prompt: str,
        max_tokens: int = 500,
        *,
        _bucket: str = "claude_extract",
        cache_prompt: bool = True,
    ) -> str:
        """Call Claude API with screenshot + prompt.

        #720 follow-up: ``cache_prompt=True`` (default) marks the
        prompt block with ``cache_control: ephemeral`` so callers
        firing the same extraction template repeatedly within the
        5-min TTL hit the cache.
        """
        import requests

        if not self.api_key:
            logger.warning("ClaudeExtractor: no API key")
            return ""

        # #518 — JPEG/PNG/WEBP per env; same dimensions as source.
        b64, media_type = encode_screenshot_for_claude(screenshot)

        # #720 — prompt block first (so cache_control caches the
        # text prefix), screenshot mutable after. Reverses the pre-PR
        # block order but Anthropic doesn't care about block ordering
        # for the same role; tested in production via the cache
        # telemetry.
        prompt_block: dict[str, Any] = {"type": "text", "text": prompt}
        if cache_prompt:
            prompt_block["cache_control"] = {"type": "ephemeral"}
        content = [
            prompt_block,
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
        ]

        t0 = time.monotonic()
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "messages": [{
                        "role": "user",
                        "content": content,
                    }],
                },
                timeout=20,
            )
            if resp.status_code != 200:
                logger.warning(
                    "ClaudeExtractor API error %s: %s",
                    resp.status_code,
                    resp.text[:500],
                )
                return ""
            payload_json = resp.json()
            credit_claude_tokens_from_response(payload_json)
            if cache_prompt:
                from .._anthropic.cache import extract_cache_telemetry
                tele = extract_cache_telemetry(payload_json)
                if tele.get("cache_read_input_tokens", 0) > 0 or tele.get(
                    "cache_creation_input_tokens", 0
                ) > 0:
                    logger.warning(
                        "  [cache] extract: read=%d created=%d input=%d output=%d",
                        tele.get("cache_read_input_tokens", 0),
                        tele.get("cache_creation_input_tokens", 0),
                        tele.get("input_tokens", 0),
                        tele.get("output_tokens", 0),
                    )
            from ..observability.claude_cost_meter import record_from_response
            record_from_response(
                source="extract_single", model=self.model, response_json=payload_json,
            )
            for block in payload_json.get("content", []):
                if block.get("type") == "text":
                    return block["text"].strip()
        except Exception as e:
            logger.warning(f"ClaudeExtractor failed: {e}")
        finally:
            _credit_claude_time(_bucket, t0)
        return ""

    def _post_anthropic_with_retry(
        self,
        payload: dict[str, Any],
        *,
        timeout: float,
        max_attempts: int = 4,
    ):
        """Back-compat shim — see :class:`AnthropicToolUseClient`.

        The implementation moved to
        :meth:`AnthropicToolUseClient.post_messages_with_retry` under
        #406. Kept here as a thin wrapper because
        ``tests/test_extractor_retry.py`` exercises it via the
        extractor surface.
        """
        return self._client.post_messages_with_retry(
            payload, timeout=timeout, max_attempts=max_attempts,
        )

    def _call_with_tool_schema(
        self,
        screenshot: Image.Image,
        prompt: str,
        *,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
        max_tokens: int = 500,
        _bucket: str = "claude_extract",
    ) -> dict | None:
        """Back-compat shim — see :meth:`AnthropicToolUseClient.call_with_tool_schema`.

        The implementation moved to the shared client under #406 so
        extraction and grounding callers share one retry policy / one
        TimeMeter bucket / one image-encoding path. This method's
        signature is preserved because tests (and a few callers
        outside this module) reach into the extractor for it.

        #720: ``cache_tools=True`` is now the default for extractor
        calls. The tool definition + JSON Schema is the largest stable
        block in any extract call (~1-3 KB for the boattrader leads
        schema, ~5 KB for nested schemas); marking it for caching cuts
        input-token cost ~30-40% on every extract call after the first
        within a 5-minute window. Was the #715 Phase 0 deferral; the
        post-merge validation showed extractor is the dominant Claude
        cost driver on every holo3-brain run, so this is the actual
        cost win.
        """
        return self._client.call_with_tool_schema(
            screenshot, prompt,
            tool_name=tool_name,
            tool_description=tool_description,
            input_schema=input_schema,
            max_tokens=max_tokens,
            time_bucket=_bucket,
            cache_tools=True,
        )

    def _call_with_tool_schema_multi(
        self,
        screenshots: list[Image.Image],
        prompt: str,
        *,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
        labels: list[str] | None = None,
        max_tokens: int = 500,
        _bucket: str = "claude_extract",
    ) -> dict | None:
        """Back-compat shim — see :meth:`AnthropicToolUseClient.call_with_tool_schema_multi`.

        #720: ``cache_tools=True`` default — see :meth:`_call_with_tool_schema`
        rationale.
        """
        return self._client.call_with_tool_schema_multi(
            screenshots, prompt,
            tool_name=tool_name,
            tool_description=tool_description,
            input_schema=input_schema,
            labels=labels,
            max_tokens=max_tokens,
            time_bucket=_bucket,
            cache_tools=True,
        )

    def _call_many(
        self,
        screenshots: list[Image.Image],
        prompt: str,
        labels: list[str] | None = None,
        max_tokens: int = 350,
        *,
        _bucket: str = "claude_extract",
        cache_prompt: bool = True,
    ) -> str:
        """Call Claude API with multiple screenshots and one prompt.

        #720 follow-up: the ``prompt`` is the stable extraction
        instruction block. Same plan + recipe = same prompt every
        call → cache_control on the first content block lets Anthropic
        cache the prompt prefix (per the 5-min ephemeral TTL).
        Screenshots and labels come AFTER the prompt and remain
        mutable. Default ON for production extract calls; opt out via
        ``cache_prompt=False`` for one-off calls where caching is pure
        overhead (no second call within TTL).
        """
        import requests

        if not self.api_key:
            logger.warning("ClaudeExtractor: no API key")
            return ""

        labels = labels or []
        prompt_block: dict[str, Any] = {"type": "text", "text": prompt}
        if cache_prompt:
            prompt_block["cache_control"] = {"type": "ephemeral"}
        content: list[dict] = [prompt_block]
        for i, screenshot in enumerate(screenshots, 1):
            label = labels[i - 1] if i - 1 < len(labels) else f"screenshot {i}"
            # #518 — JPEG/PNG/WEBP per env; same dimensions as source.
            b64, media_type = encode_screenshot_for_claude(screenshot)
            content.append({"type": "text", "text": f"Screenshot {i}: {label}"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            })

        t0 = time.monotonic()
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "messages": [{
                        "role": "user",
                        "content": content,
                    }],
                },
                timeout=30,
            )
            if resp.status_code != 200:
                logger.warning(
                    "ClaudeExtractor multi API error %s: %s",
                    resp.status_code,
                    resp.text[:500],
                )
                return ""
            payload_json = resp.json()
            credit_claude_tokens_from_response(payload_json)
            # #720 — emit cache telemetry on hits so operators can
            # audit the extract_multi cache rate in Modal logs.
            if cache_prompt:
                from .._anthropic.cache import extract_cache_telemetry
                tele = extract_cache_telemetry(payload_json)
                if tele.get("cache_read_input_tokens", 0) > 0 or tele.get(
                    "cache_creation_input_tokens", 0
                ) > 0:
                    logger.warning(
                        "  [cache] extract_many: read=%d created=%d input=%d output=%d",
                        tele.get("cache_read_input_tokens", 0),
                        tele.get("cache_creation_input_tokens", 0),
                        tele.get("input_tokens", 0),
                        tele.get("output_tokens", 0),
                    )
            from ..observability.claude_cost_meter import record_from_response
            record_from_response(
                source="extract_multi", model=self.model, response_json=payload_json,
            )
            for block in payload_json.get("content", []):
                if block.get("type") == "text":
                    return block["text"].strip()
        except Exception as e:
            logger.warning(f"ClaudeExtractor multi failed: {e}")
        finally:
            _credit_claude_time(_bucket, t0)
        return ""

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Parse JSON from response (handles markdown fences)."""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}

    def extract(self, screenshot: Image.Image) -> ExtractionResult:
        """Extract listing data from a detail page screenshot.

        When schema is set, uses dynamic prompts and populates generic
        fields; the input_schema for the tool_use call is also built
        dynamically from ``self.schema.fields``. Otherwise uses the
        legacy hardcoded prompt with the canonical marketplace shape.

        Routes through tool_use schema enforcement so the per-field
        types (string / boolean) and required-set are server-validated;
        prose-only or truncated responses become structurally
        impossible. This is the highest-leverage migration since
        ``extract`` is called once per detail page in the listings
        loop.
        """
        prompt = self._get_extract_prompt()
        input_schema = self._build_extract_input_schema()
        parsed = self._call_with_tool_schema(
            screenshot,
            prompt,
            tool_name="report_extracted_listing",
            tool_description=(
                "Report the structured fields visible on this detail page."
            ),
            input_schema=input_schema,
            max_tokens=1500,
            # #14-followup: distinct source so the per-source meter can
            # separate primary extracts from the other claude_extract
            # bucket consumers (find_content_control / verify_post_click
            # / find_filter_target) that share the default label.
            _bucket="extract_listing",
        )

        if not parsed:
            return ExtractionResult(
                raw_response="<no tool_use>", confidence=0.1, _schema=self.schema,
            )

        if self.schema:
            result = self._parse_schema_result(parsed)
            result.confidence = 0.9
            return result

        return ExtractionResult(
            year=str(parsed.get("year", "")),
            make=str(parsed.get("make", "")),
            model=str(parsed.get("model", "")),
            price=str(parsed.get("price", "")),
            phone=str(parsed.get("phone", "")),
            url=str(parsed.get("url", "")),
            seller=str(parsed.get("seller", "")),
            is_dealer=parse_bool(parsed.get("is_dealer", False)),
            raw_response=json.dumps(parsed),
            confidence=0.9,
        )

    # Type lookup for the dynamic extract input_schema. Maps the
    # ``type`` string the schema fields use to the JSON-Schema type
    # name Anthropic's tool_use expects. New types added to
    # ExtractionSchema must extend this map.
    _EXTRACT_FIELD_JSON_TYPES: ClassVar[dict[str, str]] = {
        "str": "string",
        "string": "string",
        "int": "integer",
        "integer": "integer",
        "float": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean",
    }

    def _build_extract_input_schema(self) -> dict[str, Any]:
        """Build the ``extract`` tool_use input_schema dynamically.

        - When ``self.schema`` is set: every schema field becomes a
          property; the universal ``is_spam`` boolean is appended.
          Only ``is_spam`` is server-required — every domain field
          is OPTIONAL so the brain can return partial data when a
          field isn't visible on the page (phone behind a "Show"
          button, asking_price set to "Make Offer", seller_name
          not displayed, etc.). Without this, a single missing
          field server-side-rejects the whole tool call and every
          extract step fails — produced "0 viable leads from 53
          steps" on the boattrader plan (run 20260521_064044).
          Per-field presence-checking belongs downstream where
          incomplete leads can be filtered or post-flagged, not
          at the model API boundary.
        - Otherwise: falls back to the legacy marketplace shape
          (year / make / model / price / phone / url / seller /
          is_dealer) so callers without a schema still get a
          validated response. Legacy required set kept for back
          compat — callers depending on the strict shape are
          off the schema path.
        """
        if self.schema:
            properties: dict[str, dict[str, Any]] = {}
            for field in self.schema.fields:
                name = field["name"]
                json_type = self._EXTRACT_FIELD_JSON_TYPES.get(
                    str(field.get("type", "str")).lower(), "string",
                )
                properties[name] = {"type": json_type}
            properties["is_spam"] = {"type": "boolean"}
            return {
                "type": "object",
                "properties": properties,
                # Only is_spam is required — domain fields are optional
                # so partial extractions land instead of failing.
                "required": ["is_spam"],
            }
        # Legacy / no-schema fallback shape. Every DOMAIN field is
        # optional so partial extractions land instead of failing —
        # matches the schema-path policy at lines above. Pre-fix, the
        # full 8-field required-set rejected any tool_use response
        # missing ``phone`` (universal for by-owner classifieds where
        # phone sits behind a Contact-Seller form); the extractor
        # then returned ``raw_response="<no tool_use>"`` and every
        # extract step reported 0 leads despite the page having the
        # other 7 fields visible. Only ``url`` is required (the
        # address-bar value is always readable on any loaded page);
        # the prompt explicitly tells Claude this is the one field
        # to always return.
        return {
            "type": "object",
            "properties": {
                "year": {"type": "string"},
                "make": {"type": "string"},
                "model": {"type": "string"},
                "price": {"type": "string"},
                "phone": {"type": "string"},
                "url": {"type": "string"},
                "seller": {"type": "string"},
                "is_dealer": {"type": "boolean"},
            },
            "required": ["url"],
        }

    def extract_scrolled(self, screenshot: Image.Image) -> dict:
        """Extract additional data from a scrolled-down screenshot.

        Call this after scrolling past photos to the Description section.
        Primarily looking for phone numbers and seller info.
        """
        text = self._call(screenshot, EXTRACT_SCROLLED_PROMPT)
        parsed = self._parse_json(text)
        return parsed or {}

    def extract_full(self, top_screenshot: Image.Image,
                     scrolled_screenshot: Image.Image | None = None) -> ExtractionResult:
        """Full extraction: top of page + scrolled section.

        Combines data from both screenshots for maximum coverage.
        """
        result = self.extract(top_screenshot)

        if scrolled_screenshot and not result.phone:
            extra = self.extract_scrolled(scrolled_screenshot)
            if extra.get("phone"):
                result.phone = extra["phone"]
            if extra.get("seller") and not result.seller:
                result.seller = extra["seller"]

        return result

    def _locate_description_screenshots(
        self,
        screenshots: list[Image.Image],
        labels: list[str] | None = None,
    ) -> list[int]:
        """Identify which screenshots in the multi-shot bundle show the
        Description block.

        Returns a list of indices (0-based, in the order they appear in
        ``screenshots``) that contain a recognizable Description heading
        or its body text. Empty list means "couldn't find — caller
        should fall back to the full bundle".

        Implementation: a single Haiku tool-use call that scans the
        whole bundle and returns the indices. Cost ~$0.005/listing.
        Used by ``extract_multi`` to subset the screenshots passed to
        the Sonnet phone re-extract.
        """
        if not screenshots:
            return []
        if not self.api_key:
            return []
        prompt = (
            "These are sequential viewports of one boat-listing detail "
            "page. The page has a 'Description' block: a heading "
            "labelled 'Description' followed by 1-N paragraphs of "
            "seller-written body text (a 'Show More' / 'Show Less' "
            "expand control may also be visible). Report which "
            "screenshots show ANY part of the Description block — "
            "the heading, the body text, or both. Return as a list "
            "of 1-based screenshot indices (e.g. [2,3]). If NO "
            "screenshot shows the Description block, return [].\n"
            "Do not include screenshots that show only photos, "
            "specs grid, financing widget, footer, or other "
            "non-description chrome."
        )
        try:
            parsed = self._client.call_with_tool_schema_multi(
                screenshots,
                prompt,
                tool_name="report_description_screenshots",
                tool_description="Report which screenshots show the Description block.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "1-based screenshot indices showing the Description block.",
                        },
                    },
                    "required": ["indices"],
                },
                labels=labels,
                max_tokens=80,
                time_bucket="description_locator",
                cache_tools=True,
            )
        except Exception:  # noqa: BLE001
            return []
        if not isinstance(parsed, dict):
            return []
        raw = parsed.get("indices") or []
        out: list[int] = []
        for idx in raw:
            try:
                # Convert 1-based to 0-based and bounds-check.
                i = int(idx) - 1
                if 0 <= i < len(screenshots) and i not in out:
                    out.append(i)
            except (TypeError, ValueError):
                continue
        return out

    def extract_multi(
        self,
        screenshots: list[Image.Image],
        labels: list[str] | None = None,
    ) -> ExtractionResult:
        """Extract listing data from multiple screenshots of one detail page."""
        if not screenshots:
            return ExtractionResult(confidence=0.0, _schema=self.schema)

        prompt = self._get_multi_extract_prompt()
        text = self._call_many(
            screenshots,
            prompt,
            labels=labels,
            max_tokens=450,
        )
        parsed = self._parse_json(text)

        if not parsed:
            return ExtractionResult(raw_response=text, confidence=0.1, _schema=self.schema)

        if self.schema:
            result = self._parse_schema_result(parsed)
            result.confidence = 0.9
            return result

        # Defensive parsing — accept BOTH the canonical flat shape
        # (top-level year/make/...) and the legacy nested shape
        # (``{"url": ..., "extracted": {<title-case fields>}}``) that
        # an older prompt version asked for. The legacy shape silently
        # dropped every field through ``parsed.get("year", "")`` —
        # zero leads despite the page being extractable. Unwrap the
        # ``extracted`` sub-object when present and normalize case so
        # either shape produces a populated ExtractionResult.
        nested = parsed.get("extracted") if isinstance(parsed.get("extracted"), dict) else None
        if nested:
            merged = {str(k).lower(): v for k, v in nested.items()}
            for k, v in parsed.items():
                if k != "extracted" and str(k).lower() not in merged:
                    merged[str(k).lower()] = v
            parsed = merged
        result = ExtractionResult(
            year=str(parsed.get("year", "")),
            make=str(parsed.get("make", "")),
            model=str(parsed.get("model", "")),
            price=str(parsed.get("price", "")),
            phone=str(parsed.get("phone", "")),
            url=str(parsed.get("url", "")),
            seller=str(parsed.get("seller", "")),
            is_dealer=parse_bool(parsed.get("is_dealer", False)),
            raw_response=text,
            confidence=0.9,
        )
        # WARNING-level diagnostic so production runs surface what the
        # extractor saw — Modal suppresses INFO/DEBUG, and 0-leads
        # outcomes had no visibility into whether extract_multi
        # returned populated data that got rejected downstream OR
        # returned empty data the model failed to read off the page.
        # Truncated raw_response to keep log lines bounded.
        logger.warning(
            "  [extract_multi] result: year=%r make=%r model=%r price=%r "
            "phone=%r url=%r seller=%r is_dealer=%s "
            "(nested_shape=%s, screenshots=%d, raw_len=%d)",
            result.year[:40], result.make[:40], result.model[:60],
            result.price[:40], result.phone[:40], result.url[:120],
            result.seller[:60], result.is_dealer,
            nested is not None, len(screenshots), len(text),
        )

        # Phone-only Sonnet re-extract when the primary Haiku pass left
        # the phone field empty on a detail-page extract. Gate fires if
        # *any* of year / make / model / url is populated — that's a
        # reliable signal we're on a listing detail page (not a
        # listings index, chrome-error, or CF challenge). Strict
        # all-of(year, make, model) gate was too narrow: when the
        # framework's scroll lands past the title block, Haiku misses
        # year/make and the gate doesn't fire even though we ARE on
        # the right page. Empirically Haiku misses description-embedded
        # phones at ~83% rate; Sonnet does better small-font OCR.
        # See experiments/phone_extract_fix/PLAN.md (Option A).
        if (
            os.environ.get("MANTIS_PHONE_REEXTRACT", "1") not in ("0", "false", "False")
            and not result.phone
            and (result.year or result.make or result.model or result.url)
        ):
            try:
                # Option B — focus the Sonnet re-extract on the
                # description-bearing screenshots only. The full
                # 6-viewport sweep includes the top photo carousel +
                # spec grid + financing widget + footer ads; passing
                # all of those to Sonnet (a) wastes tokens (b) gives
                # the model competing numeric content (loan amounts,
                # monthly payments, view counts) that bias digit reads.
                # A cheap Haiku locator pass returns the indices that
                # show the Description block; if it finds them, we
                # subset to those. If the locator fails or returns
                # empty, fall back to the full bundle.
                focus_screenshots = screenshots
                focus_labels = labels
                focus_indices: list[int] = []
                try:
                    focus_indices = self._locate_description_screenshots(
                        screenshots, labels=labels,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "  [extract_multi] description-locator failed: %s", exc,
                    )
                if focus_indices:
                    focus_screenshots = [screenshots[i] for i in focus_indices]
                    if labels:
                        focus_labels = [labels[i] for i in focus_indices if i < len(labels)]
                    else:
                        focus_labels = None
                logger.warning(
                    "  [extract_multi] description-locator: "
                    "indices=%s (of %d total screenshots)",
                    focus_indices, len(screenshots),
                )

                phone_prompt = (
                    "These screenshots are viewports of one "
                    "boat-listing detail page, focused on the "
                    "Description block. Look for a phone number "
                    "written by the seller (typically "
                    "embedded inside the Description body text). "
                    "Return BOTH:\n"
                    "  - phone: the digits you read, formatted as "
                    "###-###-#### or (###) ###-####. Empty string if no "
                    "phone is visible.\n"
                    "  - context_quote: the VERBATIM sentence from the "
                    "listing description that contains the phone — copied "
                    "exactly as it appears in the screenshot, including "
                    "the phone digits themselves. Empty if phone is empty.\n"
                    "Do NOT make up digits. If you can read most of a "
                    "phone but not all of it, return the empty string "
                    "rather than guessing the missing digits."
                )
                phone_parsed = self._phone_reextract_client.call_with_tool_schema_multi(
                    focus_screenshots,
                    phone_prompt,
                    tool_name="report_seller_phone",
                    tool_description="Report any phone number visible in the listing screenshots, with grounded context.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "phone": {
                                "type": "string",
                                "description": "Phone number visible in screenshots, or empty string if none.",
                            },
                            "context_quote": {
                                "type": "string",
                                "description": "Verbatim sentence from the listing containing the phone, or empty.",
                            },
                        },
                        "required": ["phone", "context_quote"],
                    },
                    labels=focus_labels,
                    max_tokens=300,
                    time_bucket="phone_reextract_sonnet",
                    cache_tools=True,
                    # M3 — bump JPEG quality from the env-resolved
                    # default (q=85) to q=95 for cleaner digit glyphs;
                    # token cost rises ~10-15% per image but materially
                    # reduces digit-position guessing on small body text.
                    image_quality=95,
                )
                call_ok = isinstance(phone_parsed, dict)
                raw_phone = ""
                raw_quote = ""
                if call_ok:
                    raw_phone = str(phone_parsed.get("phone", "") or "").strip()
                    raw_quote = str(phone_parsed.get("context_quote", "") or "").strip()
                # M1 — substring grounding check. Sonnet sometimes returns
                # a phone with hallucinated middle/tail digits (e.g.
                # ``713-598-5801`` for a real ``713-503-5091`` on the
                # same page). Requiring the phone's bare digits to
                # appear in the verbatim context_quote forces grounding
                # in actual screenshot text. Empirical 1/1 false-positive
                # caught by this gate in the 2026-06-03 TX 77304 smoke.
                accepted = ""
                reject_reason = ""
                if raw_phone:
                    phone_digits = "".join(c for c in raw_phone if c.isdigit())
                    quote_digits = "".join(c for c in raw_quote if c.isdigit())
                    if not raw_quote:
                        reject_reason = "empty_quote"
                    elif phone_digits and phone_digits in quote_digits:
                        accepted = raw_phone
                    else:
                        reject_reason = "phone_digits_not_in_quote"
                logger.warning(
                    "  [extract_multi] phone_reextract_sonnet: call_ok=%s "
                    "raw_phone=%r raw_quote=%r accepted=%r reject=%r "
                    "url=%r year=%r make=%r model=%r",
                    call_ok, raw_phone[:40], raw_quote[:120],
                    accepted[:40], reject_reason,
                    result.url[:80], result.year[:40], result.make[:40], result.model[:40],
                )
                if accepted:
                    result.phone = accepted
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "  [extract_multi] phone_reextract_sonnet failed: %s", exc,
                )

        return result

    def find_listing_content_control(self, screenshot: Image.Image) -> dict | None:
        """Find a safe expand or phone reveal control on a listing page.

        Returns a dict with x/y/action/label/reason, or None if no safe target
        is visible. This intentionally avoids generic Contact Seller forms.

        Two skip-gates protect against unsafe clicks:

        * ``schema is not None and not schema.allowed_controls`` — explicit
          empty allowlist on the active schema. Long-standing semantics.
        * ``schema is None`` — schema-less callers have no allowlist at all,
          so the safest policy is to NOT auto-click. Without this gate the
          deep-extract routine clicks whatever the model thinks is a "reveal"
          control — on real marketplace pages that often catches "Contact
          Seller", "View N Photos", or similar nav links, opening modals /
          carousels that block subsequent screenshots. Observed on the
          boattrader plan (schema-less): every extract_data step failed
          because the deep-extract loop opened the photo lightbox in
          viewport 1, then captured 5 more screenshots OF the lightbox.
        """
        if self.schema is None:
            logger.info(
                "  [content-control] no schema → no click allowlist; skipping "
                "auto-reveal. Callers that want auto-reveal must configure "
                "an ExtractionSchema with allowed_controls."
            )
            return None
        if not self.schema.allowed_controls:
            logger.info("  [content-control] schema has no allowed controls; skipping")
            return None

        debug_stem = "claude_listing_content_control"
        try:
            screenshot.save(self._debug_path(debug_stem, ".png"))
        except Exception:
            pass
        prompt = self._get_content_control_prompt()
        try:
            with open(self._debug_path(debug_stem, "_prompt.txt"), "w") as f:
                f.write(prompt)
        except Exception:
            pass

        # Force structured output via tool_use schema validation. The
        # previous _call + _parse_json path produced prose-only,
        # truncated-JSON, and tuple-style malformed JSON failures during
        # the boattrader smoke; tool_use eliminates all three by schema-
        # validating the response server-side.
        allowed_actions = list(self.schema.allowed_controls) + ["none"] if self.schema else [
            "expand_description", "show_phone", "none",
        ]
        parsed = self._call_with_tool_schema(
            screenshot,
            prompt,
            tool_name="report_content_control",
            tool_description=(
                "Report the chosen content-reveal control on the listing "
                "page (or 'none' if no safe target is visible)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "Center X in pixels"},
                    "y": {"type": "integer", "description": "Center Y in pixels"},
                    "action": {"type": "string", "enum": allowed_actions},
                    "label": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["x", "y", "action", "label", "reason"],
            },
            _bucket="find_content_control",
        )

        try:
            with open(self._debug_path(debug_stem, "_response.txt"), "w") as f:
                f.write(json.dumps(parsed) if parsed is not None else "<no tool_use>")
        except Exception:
            pass

        if not parsed:
            logger.warning("  [content-control] tool_use returned no usable result")
            return None

        action = str(parsed.get("action", "none"))
        x = int(parsed.get("x", 0))
        y = int(parsed.get("y", 0))
        if action == "none" or x == 0 or y == 0:
            return None

        label = str(parsed.get("label", ""))
        reason = str(parsed.get("reason", ""))
        forbidden = (
            "contact seller", "request info", "email seller",
            "get pre-qualified", "pre-qualified", "loan", "financing",
        )
        target_text = f"{label} {reason}".lower()
        if any(term in target_text for term in forbidden):
            logger.info(f"  [content-control] rejected unsafe target: {label[:60]}")
            return None

        return {
            "x": x,
            "y": y,
            "action": action,
            "label": label,
            "reason": reason,
        }

    def verify_post_click_navigation(
        self,
        before_screenshot: Image.Image,
        after_screenshot: Image.Image,
        intent: str,
    ) -> dict | None:
        """Decide whether a click opened detail content (URL or modal).

        Generic SPA-aware fallback for the click verifier. The URL-based
        :meth:`SiteConfig.is_detail_page` check correctly handles full-
        page-navigation sites (URL changes when a row click lands on a
        detail page). It misses single-page apps where clicks open a
        modal or overlay without changing the URL — lu.ma's event cards
        are the canonical case: ``/discover`` stays in the address bar
        but the visible content changes from a grid of cards to a single
        event detail panel.

        Calling pattern: invoke this AFTER the URL check returns False
        and BEFORE escalating to middle-click / probe-area clicks. If
        ``navigated=True``, accept the click as successful even though
        the URL didn't change. Generic primitive — no plan vocabulary,
        works for any SPA whose clicks open content in place.

        Returns a dict with keys ``navigated`` (bool), ``kind`` (one of
        ``url_change`` / ``modal`` / ``no_change`` / ``wrong_target``),
        and ``reason`` (string) — or ``None`` on API / network failure
        (caller treats None as "couldn't determine; fall through to
        existing escalation").
        """
        prompt = (
            "Compare these two screenshots taken before and after a "
            f"single click. The intent of the click was: {intent!r}.\n\n"
            "Decide whether the click successfully landed the user on "
            "detail-or-target content.\n\n"
            "Return ``navigated=true`` if EITHER:\n"
            "  • The page navigated to a different URL/page (full-page "
            "load) showing the target content, OR\n"
            "  • The same URL stayed but a modal / overlay / panel "
            "opened showing the target content (typical SPA pattern).\n\n"
            "Return ``navigated=false`` if:\n"
            "  • The page is essentially unchanged (click missed or "
            "did nothing), OR\n"
            "  • The page changed BUT to the wrong destination (login "
            "wall, error, ad, unrelated section)."
        )
        return self._call_with_tool_schema_multi(
            [before_screenshot, after_screenshot],
            prompt,
            tool_name="report_post_click_navigation",
            tool_description=(
                "Report whether a click successfully landed on detail/"
                "target content, including the case where a SPA opens a "
                "modal without changing the URL."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "navigated": {
                        "type": "boolean",
                        "description": (
                            "True iff the click landed on the intended "
                            "detail content (URL change OR same-URL modal)."
                        ),
                    },
                    "kind": {
                        "type": "string",
                        "enum": [
                            "url_change",
                            "modal",
                            "no_change",
                            "wrong_target",
                        ],
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "What you see in the AFTER frame that "
                            "confirms or denies navigation."
                        ),
                    },
                },
                "required": ["navigated", "kind", "reason"],
            },
            labels=["BEFORE click", "AFTER click"],
            _bucket="claude_verify",
        )

    # Stable across all verify_gate calls — split out so prompt caching
    # (the ``cache_tools=True`` path) actually amortises across a run.
    # Kept as ClassVar so the Anthropic prompt-cache breakpoint hashes
    # the same string every call.
    _VERIFY_GATE_TOOL_DESCRIPTION: ClassVar[str] = (
        "Report whether the gate condition holds based on what is "
        "visible in the screenshot."
    )
    _VERIFY_GATE_INPUT_SCHEMA: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "passed": {
                "type": "boolean",
                "description": "True iff the condition holds.",
            },
            "reason": {
                "type": "string",
                "description": "What you see that confirms or denies the condition.",
            },
        },
        "required": ["passed", "reason"],
    }

    def verify_gate(self, screenshot: Image.Image, expected: str) -> tuple[bool, str]:
        """Verify a gate condition from a screenshot.

        Used after setup to check if filters were actually applied.
        Returns (passed, reason). Routes through tool_use schema so the
        boolean / string shape is server-validated; any failure mode
        falls through to ``(False, reason)`` so the runner halts on
        an unverified gate (safer than silently passing).

        #421 routing:

        - **Haiku 4.5 first.** The verify client (``_verify_client``)
          defaults to Haiku — ~10× cheaper per call than Opus and
          well-capable of binary "is X visible?" judgements. The tool
          spec is cached (``cache_tools=True``) so the tool def +
          schema is amortised across all gates in a run.
        - **Opus escalation on FAIL.** If Haiku returns
          ``passed=False`` we re-ask Opus once via
          ``_verify_escalation_client``. The recovery loop a false-
          negative would trigger costs ≥ $0.50; the escalation costs
          ~$0.003 so the math is one-sided. If Opus also fails we
          return that verdict; if Opus passes we trust Opus and emit
          a WARNING so the disagreement is visible in Modal logs.
        - **Opus error keeps Haiku verdict.** If the escalation call
          errors out (None) we fall back to the Haiku verdict rather
          than hide the original failure behind a None.
        """
        prompt = (
            f"Look at this screenshot ({screenshot.width}x{screenshot.height} pixels).\n\n"
            f"Check this condition: {expected}\n\n"
            f"Look at the page heading, URL bar, result count, and any active filter tags."
        )

        parsed = self._verify_client.call_with_tool_schema(
            screenshot,
            prompt,
            tool_name="report_gate_verification",
            tool_description=self._VERIFY_GATE_TOOL_DESCRIPTION,
            input_schema=self._VERIFY_GATE_INPUT_SCHEMA,
            time_bucket="claude_verify_haiku",
            cache_tools=True,
        )

        if not parsed:
            return False, "verify_gate: tool_use returned no usable result"

        passed = bool(parsed.get("passed", False))
        reason = str(parsed.get("reason", ""))

        if not passed:
            escalated = self._verify_escalation_client.call_with_tool_schema(
                screenshot,
                prompt,
                tool_name="report_gate_verification",
                tool_description=self._VERIFY_GATE_TOOL_DESCRIPTION,
                input_schema=self._VERIFY_GATE_INPUT_SCHEMA,
                time_bucket="claude_verify_opus_escalation",
                cache_tools=True,
            )
            if escalated is not None:
                escalated_passed = bool(escalated.get("passed", False))
                escalated_reason = str(escalated.get("reason", ""))
                if escalated_passed:
                    # WARNING (not INFO) so the disagreement survives
                    # Modal's INFO-suppressed log capture — see
                    # feedback_modal_info_log_suppression. Operators
                    # auditing whether Haiku-default is regressing
                    # accuracy can grep production logs for this.
                    logger.warning(
                        "  [gate] escalation HAIKU_FAIL → OPUS_PASS: "
                        "haiku=%r opus=%r",
                        reason[:80], escalated_reason[:80],
                    )
                passed = escalated_passed
                reason = escalated_reason

        logger.info(f"  [gate] {'PASS' if passed else 'FAIL'}: {reason[:80]}")
        return passed, reason

    def find_click_target(
        self,
        screenshot: Image.Image,
        skip_count: int = 0,
        skip_urls: list[str] | None = None,
    ) -> tuple[int, int, str] | tuple[str] | None:
        """Find the next listing to click on a search results page.

        Args:
            screenshot: Current page screenshot.
            skip_count: Unused (kept for compatibility).
            skip_urls: Human-readable titles/slugs of already-extracted listings to skip.

        Returns:
            (x, y, title) — target found
            ("not_found",) — Claude confirmed no more listings
            ("error",) — API/parse failure (should retry, not treat as exhausted)
            None — empty API response (should retry)
        """
        skip_section = ""
        if skip_urls:
            skip_section = (
                "\n\nSKIP these listings (already extracted): "
                + ", ".join(skip_urls[:6])
                + "\nFind a DIFFERENT listing that is NOT in the skip list."
            )

        entity = self.schema.entity_name if self.schema else "item"
        prompt = (
            f"Look at this search-results screenshot ({screenshot.width}x{screenshot.height} pixels).\n\n"
            f"The top of the screenshot may show page header, search controls, "
            f"and filters. Result entries may start only in the LOWER part of "
            f"the screenshot; the bottom-most entry may be partially visible.\n\n"
            f"Find the first unprocessed {entity} entry — could be a card, "
            f"table row, list item, or any repeated clickable UI element. "
            f"Ignore page headers, filters, sort controls, ads, and footer links."
            f"{skip_section}\n\n"
            f"Return the CENTER coordinates of the entry's primary clickable "
            f"area (title text or main link, NOT any image).\n"
            f"If the exact title is hard to read, return approximate coordinates "
            f"and use \"unknown\" for the title.\n\n"
            f"Output ONLY valid JSON: {{\"x\": N, \"y\": N, \"title\": \"the title text or unknown\"}}\n"
            f"If no {entity} entry is visible anywhere in the screenshot, output: {{\"x\": 0, \"y\": 0, \"title\": \"none\"}}"
        )

        debug_stem = f"claude_click_skip{skip_count}"

        # Save screenshot Claude will see (for debugging)
        try:
            screenshot.save(self._debug_path(debug_stem, ".png"))
        except Exception as e:
            logger.debug(f"[claude-target] failed to save screenshot: {e}")

        try:
            with open(self._debug_path(debug_stem, "_prompt.txt"), "w") as f:
                f.write(prompt)
        except Exception as e:
            logger.debug(f"[claude-target] failed to save prompt: {e}")

        text = self._call(screenshot, prompt)

        # Save Claude's response
        try:
            with open(self._debug_path(debug_stem, "_response.txt"), "w") as f:
                f.write(text)
        except Exception as e:
            logger.debug(f"[claude-target] failed to save response: {e}")

        parsed = self._parse_json(text)

        if not parsed:
            logger.warning(f"  [claude-target] parse failed raw={text[:300]!r}")
            return ("error",)  # Parse failure — retry, don't treat as exhausted

        if parsed.get("title") == "none":
            logger.info(f"  [claude-target] Claude reported no listing for skip={skip_count}")
            return ("not_found",)  # Genuine exhaustion

        x = int(parsed.get("x", 0))
        y = int(parsed.get("y", 0))
        title = str(parsed.get("title", ""))

        if x == 0 and y == 0:
            logger.warning(f"  [claude-target] zero coordinates raw={text[:300]!r}")
            return ("error",)  # Bad coordinates — retry

        logger.info(f"  [claude-target] '{title[:40]}' at ({x}, {y})")
        return (x, y, title)

    def find_all_listings(
        self,
        screenshot: Image.Image,
        already_clicked: list[str] | None = None,
    ) -> list[tuple[int, int, str]] | tuple[str]:
        """Find ALL listing cards on the page in ONE Claude call.

        Returns:
            list[(x, y, title)] — visible listings found
            ("empty",) — screenshot looks like a normal page but no listings visible
            ("blocked",) — screenshot appears to be an error/block/rate-limit page
            ("error",) — API/parse failure

        Cost: ~$0.003-0.005 (one call regardless of how many cards).
        The caller clicks each sequentially without calling Claude again.

        ``already_clicked``: titles of listings the runner has already
        clicked-through in prior loop iterations. When non-empty, the
        prompt explicitly instructs the brain NOT to return them — saves
        Claude tokens (smaller scan output) and shifts dedup from a
        post-hoc fuzzy title-substring match in ``click.py`` to the
        model's classification step where it's both cheaper and cleaner.
        Caller passes the bounded tail (last ~12) so the prompt token
        budget stays small on long-running loops. See issue #597.
        """
        debug_stem = "claude_find_all"
        # All entity / spam / layout context comes from the schema. When no
        # schema is provided, the prompt stays entity-neutral — it asks for
        # "the items the user wants to click" without naming a domain.
        entity = self.schema.entity_name if self.schema else "item"
        spam_label = self.schema.spam_label if self.schema else "non-organic"
        spam_examples_clause = ""
        if self.schema and self.schema.spam_indicators:
            examples = ", ".join(f'"{s}"' for s in self.schema.spam_indicators[:6])
            spam_examples_clause = (
                f"\nSpam signals to filter (caller-provided): {examples}."
            )
        already_clicked_clause = ""
        if already_clicked:
            # Cap defensively in case the caller forgets to slice; the
            # tail-12 convention keeps the prompt bounded.
            tail = list(already_clicked)[-12:]
            tail_str = ", ".join(f'"{t}"' for t in tail if t)
            if tail_str:
                already_clicked_clause = (
                    f"\nALREADY CLICKED — DO NOT RETURN (these entries "
                    f"are already extracted by the runner; returning them "
                    f"wastes a click cycle): {tail_str}."
                )
        prompt = (
            f"Look at this screenshot ({screenshot.width}x{screenshot.height} pixels).\n\n"
            f"Find ALL eligible {entity} entries visible in this screenshot. "
            f"Entries may be cards-with-photos, table rows, list items, profile "
            f"rows, panel sections, or any repeated UI element on a results "
            f"page. Include the bottom-most entry even if partially visible.\n\n"
            f"STRICT FILTER: only return organic entries the user can click. "
            f"Skip {spam_label} content, sponsored ads, navigation tabs, "
            f"sort/filter controls, headers, footers, and pagination bars."
            f"{spam_examples_clause}"
            f"{already_clicked_clause}\n\n"
            f"If the screenshot shows an error page, rate limit, bot check, "
            f"CAPTCHA, or sign-in wall, mark it as blocked.\n\n"
            f"Return the CENTER coordinates of each entry's primary clickable "
            f"area (the title text or the row's main link, not any image). "
            f"If a title is hard to read, use \"unknown\".\n\n"
            f"Also check: is there a pagination control (page numbers or a "
            f"Next button) visible at the bottom? Note its Y coordinate.\n\n"
            f"Output ONLY valid JSON:\n"
            f"{{\"status\": \"ok\", \"listings\": [{{\"x\": N, \"y\": N, \"title\": \"text or unknown\", "
            f"\"is_organic\": true/false, \"reason\": \"brief\"}}, ...], "
            f"\"pagination_y\": N_or_null}}\n"
            f"If this looks like a normal page but no entries are visible, output:\n"
            f"{{\"status\": \"empty\", \"listings\": []}}\n"
            f"If the screenshot appears blocked or errored, output:\n"
            f"{{\"status\": \"blocked\", \"listings\": []}}"
        )

        try:
            screenshot.save(self._debug_path(debug_stem, ".png"))
        except Exception as e:
            logger.debug(f"[find_all] failed to save screenshot: {e}")

        try:
            with open(self._debug_path(debug_stem, "_prompt.txt"), "w") as f:
                f.write(prompt)
        except Exception as e:
            logger.debug(f"[find_all] failed to save prompt: {e}")

        text = self._call(screenshot, prompt, max_tokens=1500)

        try:
            with open(self._debug_path(debug_stem, "_response.txt"), "w") as f:
                f.write(text)
        except Exception as e:
            logger.debug(f"[find_all] failed to save response: {e}")

        # Parse JSON payload
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        parsed: dict | list | None = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            pass

        if parsed is None:
            import re as _re
            match = _re.search(r'\{.*\}', text, _re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    parsed = None
            else:
                match = _re.search(r'\[.*\]', text, _re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group())
                    except json.JSONDecodeError:
                        parsed = None

        if parsed is None:
            logger.warning(f"  [find_all] No parseable JSON in response: {text[:200]}")
            return ("error",)

        status = "ok"
        items: list = []
        if isinstance(parsed, dict):
            status = str(parsed.get("status", "ok")).lower()
            raw_items = parsed.get("listings", [])
            if isinstance(raw_items, list):
                items = raw_items
        elif isinstance(parsed, list):
            # Backward compatibility with the old prompt.
            items = parsed
        else:
            logger.warning(f"  [find_all] Unexpected payload type: {type(parsed).__name__}")
            return ("error",)

        if status == "blocked":
            logger.warning("  [find_all] Claude classified screenshot as blocked/error page")
            return ("blocked",)
        if status == "empty":
            logger.info("  [find_all] Claude reported no listings in this viewport")
            return ("empty",)

        results = []
        for item in items:
            x = int(item.get("x", 0))
            y = int(item.get("y", 0))
            title = str(item.get("title", "unknown"))
            seller = str(item.get("seller", ""))
            reason = str(item.get("reason", ""))
            is_private_seller = item.get("is_private_seller")
            is_dealer = parse_bool(item.get("is_dealer", False))
            is_sponsored = parse_bool(item.get("is_sponsored", False))
            card_text = f"{title} {seller} {reason}"
            if (
                is_private_seller is False
                or is_dealer
                or is_sponsored
                or contains_dealer_text(card_text)
                or seller_looks_like_dealer(seller)
            ):
                logger.info(
                    "  [find_all] Skipping non-private card: title='%s' seller='%s' reason='%s'",
                    title[:50],
                    seller[:50],
                    reason[:80],
                )
                continue
            if x > 0 and y > 0:
                results.append((x, y, title))

        logger.info(f"  [find_all] Found {len(results)} listings in one call")
        return results

    def find_paginate_target(self, screenshot: Image.Image) -> tuple[int, int, str] | tuple[str] | None:
        """Find the Next page button or next page number on a search results page.

        Returns:
            (x, y, label) — pagination target found
            ("not_found",) — Claude confirmed no next-page control is visible
            ("error",) — API/parse failure or unusable coordinates
            None — empty response
        """
        prompt = (
            f"Look at this results-page screenshot ({screenshot.width}x{screenshot.height} pixels).\n\n"
            f"You are near the bottom of the results. The pagination bar is usually BELOW the last result cards "
            f"and ABOVE the footer/social icons. Find the control that goes to the NEXT results page.\n\n"
            f"Valid targets:\n"
            f"- A link that says 'Next'\n"
            f"- A right-arrow or chevron like '>'\n"
            f"- The next page number to the right of the current page number\n\n"
            f"Do NOT return footer links, social icons, newsletter/signup buttons, or ad links.\n\n"
            f"Return the CENTER coordinates of the next-page control.\n"
            f"Output ONLY valid JSON: {{\"x\": N, \"y\": N, \"label\": \"Next or 2 or >\"}}\n"
            f"If no next-page control is visible anywhere in the screenshot: {{\"x\": 0, \"y\": 0, \"label\": \"none\"}}"
        )

        debug_stem = "claude_paginate"

        try:
            screenshot.save(self._debug_path(debug_stem, ".png"))
        except Exception as e:
            logger.debug(f"[claude-paginate] failed to save screenshot: {e}")

        try:
            with open(self._debug_path(debug_stem, "_prompt.txt"), "w") as f:
                f.write(prompt)
        except Exception as e:
            logger.debug(f"[claude-paginate] failed to save prompt: {e}")

        text = self._call(screenshot, prompt)

        try:
            with open(self._debug_path(debug_stem, "_response.txt"), "w") as f:
                f.write(text)
        except Exception as e:
            logger.debug(f"[claude-paginate] failed to save response: {e}")

        parsed = self._parse_json(text)

        if not parsed:
            logger.warning(f"  [claude-paginate] parse failed raw={text[:300]!r}")
            return ("error",)

        if parsed.get("label") == "none":
            logger.info("  [claude-paginate] Claude reported no next-page control visible")
            return ("not_found",)

        x = int(parsed.get("x", 0))
        y = int(parsed.get("y", 0))
        label = str(parsed.get("label", ""))

        if x == 0 and y == 0:
            logger.warning(f"  [claude-paginate] zero coordinates raw={text[:300]!r}")
            return ("error",)

        logger.info(f"  [claude-paginate] '{label[:20]}' at ({x}, {y})")
        return (x, y, label)

    def find_filter_target(
        self,
        screenshot: Image.Image,
        filter_intent: str,
    ) -> dict | None:
        """Find a filter element on the page and determine how to interact with it.

        Args:
            screenshot: Current page screenshot.
            filter_intent: Description like "Click Private Seller option in seller type filter"
                          or "Enter zip code 33101 in location search field".

        Returns:
            dict with keys: x, y, action ("click"|"type"|"select"), value (for type/select),
                           label (what was found)
            None on failure.
        """
        prompt = (
            f"Look at this screenshot ({screenshot.width}x{screenshot.height} pixels).\n\n"
            f"TASK: {filter_intent}\n\n"
            f"This is a search/results page. Look for the matching filter "
            f"control wherever it lives — left sidebar, right sidebar, top "
            f"filter bar, modal, or inline above the results.\n\n"
            f"Common control shapes:\n"
            f"- Checkboxes or radio buttons next to a text label\n"
            f"- Text input fields with a label nearby or placeholder text\n"
            f"- Dropdown menus (current selection visible with a chevron)\n"
            f"- Clickable text links / pills for filter options\n"
            f"- Sliders or range inputs\n\n"
            f"The exact label in the TASK may differ from what's on screen — "
            f"prefer semantic matches over exact strings (e.g. a 'Sort by' "
            f"control could be labelled 'Order by' on this site). Match by "
            f"the FUNCTION the TASK describes, not literal wording.\n\n"
            f"Determine the interaction:\n"
            f"- \"click\" for checkboxes, radio buttons, links, pills, toggle buttons\n"
            f"- \"type\" for text input fields (click field, clear, type value)\n"
            f"- \"select\" for dropdown menus (click to open, then pick option)\n\n"
            f"Return the CENTER coordinates of the interactive element.\n"
            f"Output ONLY valid JSON:\n"
            f"{{\"x\": N, \"y\": N, \"action\": \"click|type|select\", "
            f"\"value\": \"text to type or option to select or empty\", "
            f"\"label\": \"what element you found\"}}\n"
            f"If you cannot find any matching filter element anywhere on the page:\n"
            f"{{\"x\": 0, \"y\": 0, \"action\": \"not_found\", "
            f"\"value\": \"\", \"label\": \"describe what you see instead\"}}"
        )

        debug_stem = "claude_filter"
        try:
            screenshot.save(self._debug_path(debug_stem, ".png"))
        except Exception:
            pass
        try:
            with open(self._debug_path(debug_stem, "_prompt.txt"), "w") as f:
                f.write(prompt)
        except Exception:
            pass

        # tool_use schema enforcement — the prompt-only "Output ONLY
        # valid JSON" pattern produced prose-only / truncated /
        # malformed responses on the boattrader smoke. Schema-validated
        # tool_use makes those failure modes structurally impossible.
        parsed = self._call_with_tool_schema(
            screenshot,
            prompt,
            tool_name="report_filter_target",
            tool_description=(
                "Report the chosen filter control's coordinates and how "
                "to interact with it (click / type / select), or "
                "not_found if no matching element is visible."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "Center X in pixels"},
                    "y": {"type": "integer", "description": "Center Y in pixels"},
                    "action": {
                        "type": "string",
                        "enum": ["click", "right_click", "type", "select", "not_found"],
                    },
                    "value": {"type": "string"},
                    "label": {"type": "string"},
                },
                "required": ["x", "y", "action", "value", "label"],
            },
            _bucket="find_filter_target",
        )

        try:
            with open(self._debug_path(debug_stem, "_response.txt"), "w") as f:
                f.write(json.dumps(parsed) if parsed is not None else "<no tool_use>")
        except Exception:
            pass

        if not parsed:
            logger.warning("  [claude-filter] tool_use returned no usable result")
            return None

        if parsed.get("action") == "not_found":
            label = parsed.get("label", "unknown")
            logger.info(f"  [claude-filter] Element not visible: {filter_intent[:50]}")
            logger.info(f"  [claude-filter] What Claude sees: {label[:100]}")
            print(f"  [claude-filter] NOT FOUND: {label[:100]}")
            return None

        x = int(parsed.get("x", 0))
        y = int(parsed.get("y", 0))
        if x == 0 and y == 0:
            logger.warning("  [claude-filter] zero coordinates")
            return None

        result = {
            "x": x,
            "y": y,
            "action": str(parsed.get("action", "click")),
            "value": str(parsed.get("value", "")),
            "label": str(parsed.get("label", "")),
        }
        logger.info(f"  [claude-filter] '{result['label'][:40]}' at ({x},{y}) action={result['action']}")
        return result

    @property
    def _form_target_provider(self):
        """Lazily-constructed :class:`ClaudeFormTargetProvider` (#406).

        The form-target grounding methods moved out of this class into
        :mod:`mantis_agent.form_targeting.claude`. The methods below
        are kept as back-compat shims so existing callers (tests,
        ad-hoc scripts) still work; production code should use the
        provider on :class:`StepContext` instead.
        """
        # #434 follow-up: this property used double-underscore attribute
        # name (``__form_target_provider_cached``) for the cache, which
        # gets name-mangled to ``_ClassName__form_target_provider_cached``
        # in attribute SET but is read back via ``getattr`` with the
        # literal pre-mangled name — so the cache never hit and every
        # access reconstructed the provider. Before #434 this was
        # invisible because the new provider always shared the
        # extractor's ``_client`` (which is what test mocks pinned), so
        # the leak only matters now that the form-target client is a
        # distinct instance.
        prov = getattr(self, "_form_target_provider_cached", None)
        if prov is None:
            from ..form_targeting.claude import ClaudeFormTargetProvider

            # #434: split the form-target client off the main extractor
            # client so grounding can run on a cheaper model
            # (``form_target_model``, default Haiku) while extraction
            # stays on the heavier one. Same retry policy since both
            # clients are :class:`AnthropicToolUseClient`.
            form_target_client = AnthropicToolUseClient(
                api_key=self.api_key,
                model=self.form_target_model,
                log_prefix="ClaudeFormTarget",
            )
            prov = ClaudeFormTargetProvider(form_target_client)
            self._form_target_provider_cached = prov
        return prov

    def find_form_target(
        self,
        screenshot: Image.Image,
        intent: str,
        *,
        target_label: str = "",
        target_value: str = "",
        target_aliases: list[str] | None = None,
    ) -> dict | None:
        """Back-compat shim — see :meth:`ClaudeFormTargetProvider.find_form_target`.

        Implementation moved under #406; this method delegates so
        existing callers keep working without touching every test
        and step-handler call site at once. Callers should migrate
        to use the provider on :class:`StepContext` directly.
        """
        return self._form_target_provider.find_form_target(
            screenshot, intent,
            target_label=target_label,
            target_value=target_value,
            target_aliases=target_aliases,
        )

    def verify_dropdown_value(
        self,
        screenshot: Image.Image,
        dropdown_label: str,
        expected_value: str,
    ) -> dict | None:
        """Read the current displayed value of a (closed) dropdown and
        confirm it matches ``expected_value``.

        Used by the ``select_option`` form-handler branch as a post-
        click validation step. After the runner clicks an option in
        an open dropdown menu, the menu closes and the dropdown shows
        the picked value. This method screenshots that state and
        asks Claude what the dropdown reads — if the answer
        differs from the requested option, the runner knows the
        click landed on a different option (canonical case: y-
        coordinate disambiguation between adjacent menu items, where
        the runner reported ``select:Priority=High`` but the dropdown
        actually committed ``Critical``).

        **Debiased prompt design.** The LLM is *not* told what value
        the runner expected — telling it primes the answer ("expected
        High, looks like High to me"). Instead, ask only for the
        currently-displayed value; compute the semantic match locally
        below. This ablation matters: in the priority-plan smoke run,
        the biased version reported ``matches=True / observed=High``
        while the saved page showed ``Critical``, defeating the
        whole point of the post-click validator.

        Returns a dict::

            {"matches": bool, "observed": str}

        ``matches`` is True iff the observed value semantically equals
        the expected value (case-insensitive substring on either side
        — the dropdown might render a label that's a substring of the
        canonical option, or vice versa for verbose UIs). ``observed``
        is the literal value Claude read from the dropdown — surfaced
        in logs / mismatch errors so operators can see what actually
        committed.

        Returns ``None`` on any API failure — caller should treat as
        "could not verify; trust the click happened" rather than
        forcing a retry on every API blip.
        """
        return self._form_target_provider.verify_dropdown_value(
            screenshot, dropdown_label, expected_value,
        )

    @staticmethod
    def _semantic_dropdown_match(observed: str, expected: str) -> bool:
        """Back-compat shim — see
        :func:`mantis_agent.form_targeting.claude._semantic_dropdown_match`.
        """
        from ..form_targeting.claude import _semantic_dropdown_match
        return _semantic_dropdown_match(observed, expected)

    def find_target_by_affordance(
        self,
        screenshot: Image.Image,
        intent: str,
    ) -> dict | None:
        """Back-compat shim — see
        :meth:`ClaudeFormTargetProvider.find_target_by_affordance`.
        """
        return self._form_target_provider.find_target_by_affordance(
            screenshot, intent,
        )
