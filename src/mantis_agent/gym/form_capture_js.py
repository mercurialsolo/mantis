"""JS payloads for form-field capture + replay (epic #358 Phase B).

Lives in its own module so xdotool / Playwright env adapters share
one implementation. The two-half pattern — single JS expression on
capture, single JS expression on replay — keeps the env adapters
thin (one ``cdp_evaluate`` or ``page.evaluate`` call each).

Capture skips passwords by default (records the field key, masks
the value, sets ``masked=true``) so secrets never land in
``PauseState`` JSON. ``MANTIS_PAUSE_CAPTURE_PASSWORDS=1`` opts in
for test / debug only.

Replay is best-effort: selectors that no longer match (the DOM
shifted between pause and resume) are silently skipped — never
fail a resume because the page changed shape.
"""

from __future__ import annotations

import os


def capture_js() -> str:
    """Return a JS expression that evaluates to a list of captured
    form fields, each ``{selector, kind, value, masked}``.

    The expression is wrapped in an IIFE so the env's eval call
    receives a clean value. Selectors prefer ``data-*`` > ``id``
    > shortest ascending CSS path — keeps the snapshot robust
    against minor DOM churn between pause and resume.
    """
    capture_passwords = (
        os.environ.get("MANTIS_PAUSE_CAPTURE_PASSWORDS", "").lower()
        in ("1", "true", "yes")
    )
    capture_pw_literal = "true" if capture_passwords else "false"
    return (
        "(() => {"
        # ── stable-selector heuristic ────────────────────────────
        "const stableSelector = (el) => {"
        "  for (const attr of el.attributes || []) {"
        "    if (attr.name.startsWith('data-') && attr.value) {"
        "      return `[${attr.name}=${JSON.stringify(attr.value)}]`;"
        "    }"
        "  }"
        "  if (el.id) return `#${CSS.escape(el.id)}`;"
        # Ascending path fallback — name + nth-of-type up to 4 levels.
        "  const parts = [];"
        "  let cur = el;"
        "  for (let depth = 0; depth < 4 && cur && cur !== document.body; depth++) {"
        "    let s = cur.tagName.toLowerCase();"
        "    if (cur.classList && cur.classList.length) {"
        "      s += '.' + [...cur.classList].slice(0, 2).map(c => CSS.escape(c)).join('.');"
        "    }"
        "    const parent = cur.parentElement;"
        "    if (parent) {"
        "      const sameTag = [...parent.children].filter(c => c.tagName === cur.tagName);"
        "      if (sameTag.length > 1) {"
        "        s += `:nth-of-type(${sameTag.indexOf(cur) + 1})`;"
        "      }"
        "    }"
        "    parts.unshift(s);"
        "    cur = parent;"
        "  }"
        "  return parts.join(' > ');"
        "};"
        # ── field classification + value extraction ─────────────
        "const classify = (el) => {"
        "  const tag = el.tagName.toLowerCase();"
        "  if (tag === 'select') return 'select';"
        "  if (tag === 'textarea') return 'text';"
        "  if (el.isContentEditable) return 'contenteditable';"
        "  const type = (el.type || 'text').toLowerCase();"
        "  if (type === 'checkbox') return 'checkbox';"
        "  if (type === 'radio') return 'radio';"
        "  return 'text';"
        "};"
        "const extractValue = (el, kind) => {"
        "  if (kind === 'checkbox' || kind === 'radio') return el.checked ? 'true' : 'false';"
        "  if (kind === 'contenteditable') return el.textContent || '';"
        "  return el.value || '';"
        "};"
        # ── walk + filter ────────────────────────────────────────
        f"const capturePasswords = {capture_pw_literal};"
        "const sel = "
        "  \"input:not([type='hidden']), select, textarea, [contenteditable='true']\";"
        "const out = [];"
        "for (const el of document.querySelectorAll(sel)) {"
        "  const type = (el.type || '').toLowerCase();"
        "  const isPassword = type === 'password';"
        "  const kind = classify(el);"
        "  const selector = stableSelector(el);"
        "  if (!selector) continue;"
        "  if (isPassword && !capturePasswords) {"
        "    out.push({selector, kind: 'text', value: '', masked: true});"
        "  } else {"
        "    out.push({selector, kind, value: extractValue(el, kind), masked: false});"
        "  }"
        "}"
        "return out;"
        "})()"
    )


def replay_js(serialized_entries: str) -> str:
    """Return a JS expression that walks the captured entries (a
    pre-stringified JSON array) and re-applies each one. Missing
    selectors silently skipped; ``masked`` entries skipped (the
    caller re-prompts).

    Returns an object ``{applied, skipped, missing}`` for the caller
    to log how the replay went.
    """
    return (
        "(() => {"
        f"const entries = {serialized_entries};"
        "let applied = 0, skipped = 0, missing = 0;"
        "for (const entry of entries) {"
        "  if (entry.masked) { skipped++; continue; }"
        "  let el;"
        "  try { el = document.querySelector(entry.selector); }"
        "  catch (e) { el = null; }"
        "  if (!el) { missing++; continue; }"
        "  try {"
        "    if (entry.kind === 'text') {"
        "      el.value = entry.value;"
        "      el.dispatchEvent(new Event('input', {bubbles: true}));"
        "      el.dispatchEvent(new Event('change', {bubbles: true}));"
        "    } else if (entry.kind === 'contenteditable') {"
        "      el.textContent = entry.value;"
        "      el.dispatchEvent(new Event('input', {bubbles: true}));"
        "    } else if (entry.kind === 'checkbox' || entry.kind === 'radio') {"
        "      el.checked = entry.value === 'true';"
        "      el.dispatchEvent(new Event('change', {bubbles: true}));"
        "    } else if (entry.kind === 'select') {"
        "      el.value = entry.value;"
        "      el.dispatchEvent(new Event('change', {bubbles: true}));"
        "    }"
        "    applied++;"
        "  } catch (e) { skipped++; }"
        "}"
        "return {applied, skipped, missing};"
        "})()"
    )


__all__ = ["capture_js", "replay_js"]
