"""Modal-side ``MicroPlanRunner`` host — full mantis stack inside a container.

The local ``mantis plan run`` CLI runs Playwright on the dev laptop and
talks to a remote brain. That works for unit-of-work testing but it
doesn't reproduce the host integration's production architecture, where
the BROWSER also runs inside a Modal container under Xvfb so the
chromium-detection signals (no DISPLAY, missing GPU, ``navigator.
webdriver``, …) that Cloudflare keys on are absent.

This module deploys ``mantis-plan-runner`` — a Modal app exposing a
single function that constructs ``XdotoolGymEnv`` (Xvfb-backed real
Chromium) + ``Holo3Brain`` (HTTP to the brain endpoint) +
``ClaudeGrounding`` + ``ClaudeExtractor``, runs ``MicroPlanRunner``
end-to-end, and returns the same result dict the local CLI writes to
``result.json``. The local ``mantis plan run-modal`` subcommand
submits the plan and renders the same per-step rollup the local CLI
prints.

## Deploy

    uv run modal deploy deploy/modal/modal_plan_runner.py

That gives you an app named ``mantis-plan-runner`` whose ``run_plan``
function is callable via ``modal.Function.lookup`` from any local
process. The image bakes in xvfb, xdotool, scrot, chromium, and the
``mantis_agent`` package; first cold-start is ~60 s after deploy.

## Required Modal Secret

Mount a Modal Secret named ``mantis-plan-runner-secrets`` containing:

    ANTHROPIC_API_KEY=sk-ant-...
    MANTIS_API_TOKEN=...                # optional — used as the
                                         # X-Mantis-Token header on
                                         # the brain endpoint when no
                                         # caller-supplied headers
    OXYLABS_USERNAME=...                # optional — only consulted
    OXYLABS_PASSWORD=...                # when ``use_proxy=True``
    OXYLABS_ENTRYPOINT=pr.oxylabs.io:10000
    OXYLABS_COUNTRY=US
    OXYLABS_STATE=florida               # optional
    OXYLABS_CITY=miami                  # optional

Create / update::

    modal secret create mantis-plan-runner-secrets \\
        ANTHROPIC_API_KEY=sk-ant-... \\
        MANTIS_API_TOKEN=...

## Calling from the CLI

    uv run mantis plan run-modal plans/staff-crm.txt \\
        --endpoint https://workspace--mantis-server-api.modal.run/v1 \\
        --header X-Mantis-Token=... \\
        --output-dir outputs/staff-crm-modal-validation

The CLI is a thin remote driver — same args as ``mantis plan run``,
plus ``--app-name`` (default ``mantis-plan-runner``) to override the
deployed function name.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import modal


APP_NAME = "mantis-plan-runner"

# Lightweight image: just what XdotoolGymEnv + ClaudeExtractor +
# ClaudeGrounding + Holo3Brain need. No CUDA, no llama.cpp, no
# OSWorld — those live in the brain-side image (modal_mantis_server).
runner_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        # Browser + automation tools (XdotoolGymEnv requires these).
        "xvfb",
        "xdotool",
        "scrot",
        "chromium",
        "chromium-driver",
        "x11-utils",
        # Window manager + panel for a normal desktop context. Without
        # a WM, Chromium's window has no decorations / focus handling
        # and Cloudflare's bot-score classifier flags the fingerprint
        # as automation-driven (the challenge page never auto-resolves).
        "openbox",
        "tint2",
        # CDP read for current-URL — used by _runner_helpers.read_current_url.
        "curl",
        # Local HTTP proxy with upstream auth: Chromium's --proxy-server flag
        # does not reliably honor embedded user:pass@ credentials, so we run
        # tinyproxy inside the container, point Chrome at 127.0.0.1:8888
        # (no auth), and let tinyproxy hold the Oxylabs auth and forward.
        "tinyproxy",
        # Fonts for realistic rendering (Cloudflare flags missing fonts).
        "fonts-liberation",
        "fonts-noto-color-emoji",
        "libnss3",
        "libxss1",
        "libasound2",
    )
    .pip_install(
        # Pinned to the orchestrator extras + a few extras XdotoolGymEnv pulls.
        "pillow>=10.0",
        "requests>=2.28",
        "pydantic>=2",
        "anthropic>=0.34",
        "mss>=9.0",
        "python-dotenv",
        # #509: per-run Augur DebugSession bundle + optional live streaming.
        # 0.1.2+ fires an immediate session-opened heartbeat for faster
        # workspace badge updates.
        "augur-sdk>=0.3.0,<0.4",
    )
    .add_local_python_source("mantis_agent")
    .add_local_dir(
        "src/mantis_agent/prompts/files",
        remote_path="/root/mantis_agent/prompts/files",
    )
)

app = modal.App(APP_NAME)


# ── Helper: bring up Xvfb on :99 once per container ──────────────────


_XVFB_DISPLAY = ":99"
_XVFB_VIEWPORT = (1280, 720)


def _ensure_xvfb_running() -> None:
    """Start Xvfb on the canonical display if it isn't already.

    Modal containers don't ship with X. ``XdotoolGymEnv`` can spawn
    its own Xvfb (when ``display=None``) but inside a function we
    benefit from spinning it up once and reusing across calls in
    the same warm container. The display number is fixed at ``:99``
    by convention.
    """
    # Fast path: another invocation in this container already booted Xvfb.
    if os.environ.get("DISPLAY") == _XVFB_DISPLAY and _xvfb_alive():
        return

    width, height = _XVFB_VIEWPORT
    subprocess.Popen(
        [
            "Xvfb",
            _XVFB_DISPLAY,
            "-screen", "0", f"{width}x{height}x24",
            "-ac",
            "+extension", "RANDR",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = _XVFB_DISPLAY
    # Wait briefly for Xvfb to come up — xdpyinfo confirms.
    deadline = time.time() + 10
    while time.time() < deadline:
        if _xvfb_alive():
            break
        time.sleep(0.2)
    else:
        raise RuntimeError(
            f"Xvfb did not come up on {_XVFB_DISPLAY} within 10s — "
            "check that xvfb is installed in the image."
        )

    # Once Xvfb is alive, bring up a window manager so Chromium has a
    # normal desktop context — Cloudflare's bot-score classifier looks
    # at window decoration / focus signals; without a WM the browser
    # presents as automation-driven and the JS challenge never
    # auto-resolves. Both spawn detached and inherit DISPLAY=:99 from
    # the env we just set.
    subprocess.Popen(
        ["openbox"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.Popen(
        ["tint2"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Brief settle so WM grabs the X server before Chromium starts.
    time.sleep(0.5)


def _xvfb_alive() -> bool:
    try:
        proc = subprocess.run(
            ["xdpyinfo", "-display", _XVFB_DISPLAY],
            capture_output=True,
            timeout=3,
        )
        return proc.returncode == 0
    except Exception:
        return False


# ── Helper: spawn tinyproxy as a local auth-holder ──────────────────


_TINYPROXY_PORT = 8888
_TINYPROXY_CONFIG = "/tmp/tinyproxy.conf"
_TINYPROXY_LOG = "/tmp/tinyproxy.log"
_TINYPROXY_PID = "/tmp/tinyproxy.pid"


def _build_oxylabs_username(session: str = "mantis", *, geo: bool = False) -> str:
    """Compose the ``customer-USER[-cc-…]-sessid-S`` username.

    Reads ``OXYLABS_USERNAME`` and the optional country/state/city/
    session pins. Returns an empty string when no username is set —
    callers should treat that as "no proxy" and bring the browser up
    on direct egress.

    The ``geo`` flag controls whether the ``cc-/st-/city-`` slots are
    appended. The default (``False``) is intentional: Oxylabs returns
    502 on tunnel CONNECT when the user's plan doesn't support
    city-level pinning on the standard residential entrypoint
    (``pr.oxylabs.io:10000``). City-level pinning lives on
    ``pr.oxylabs.io:7777`` (``OXYLABS_CITY_ENTRYPOINT``) and only
    works for plans that explicitly support it. Diagnose-then-enable.

    **State format** — Oxylabs docs require US states to be
    prefixed with the country code: ``us_california``, not
    ``california``. Canonical example: ``st-us_california-city-los_angeles``.
    When ``OXYLABS_STATE`` lacks the ``<cc>_`` prefix, this function
    prepends ``cc.lower() + "_"`` automatically — so operators can
    set ``OXYLABS_STATE=florida`` and the helper produces the
    canonical ``st-us_florida``. Without this prefix, Oxylabs treats
    the state as unknown and silently falls back to random global
    rotation (observed: ``st-florida`` returned Ukraine and Brazil
    IPs in production diagnostic).

    **City format** — bare lowercase, with spaces replaced by
    underscores: ``city-los_angeles`` (not ``city-Los Angeles``).
    """
    raw_user = os.environ.get("OXYLABS_USERNAME", "").strip()
    if not raw_user:
        return ""

    parts = [f"customer-{raw_user}"]
    if geo:
        cc = os.environ.get("OXYLABS_COUNTRY", "").strip()
        state = os.environ.get("OXYLABS_STATE", "").strip()
        city = os.environ.get("OXYLABS_CITY", "").strip()

        if cc:
            parts += ["cc", cc]
        if state:
            # Oxylabs convention: state names are country-prefixed
            # ("us_california"). If the operator wrote "florida"
            # without prefix, attach the country code lowercased so
            # we emit the canonical "us_florida" form. Skip the
            # auto-prefix if the value already contains "_" (operator
            # already wrote the canonical form).
            if "_" in state:
                formatted_state = state.lower()
            elif cc:
                formatted_state = f"{cc.lower()}_{state.lower()}"
            else:
                formatted_state = state.lower()
            parts += ["st", formatted_state]
        if city:
            # Bare lowercase, spaces → underscores.
            parts += ["city", city.lower().replace(" ", "_")]
    if session:
        parts += ["sessid", session]
    return "-".join(parts)


def _resolve_upstream_proxy() -> tuple[str, str, str] | None:
    """Pick the upstream proxy provider for tinyproxy to forward to.

    Returns ``(username, password, host_port)`` tuple ready to splice
    into the tinyproxy ``upstream`` directive, or ``None`` if no
    provider creds are mounted.

    Selection logic, in order:
    1. ``MANTIS_PROXY_PROVIDER`` env override (``oxylabs`` / ``privateproxy``)
    2. PrivateProxy when its three vars are set (more reliable in our
       testing — BoatTrader's Cloudflare flags Oxylabs residential pool
       IPs as bot traffic, returning 403 on CONNECT)
    3. Oxylabs fallback with bare ``customer-USER`` (no geo pins —
       city-level pinning needs a different entrypoint and plan)
    """
    pp_user = os.environ.get("PRIVATEPROXY_USERNAME", "").strip()
    pp_pass = os.environ.get("PRIVATEPROXY_PASSWORD", "").strip()
    pp_entry = os.environ.get("PRIVATEPROXY_ENTRYPOINT", "").strip()
    pp_complete = bool(pp_user and pp_pass and pp_entry)

    ox_pass = os.environ.get("OXYLABS_PASSWORD", "").strip()
    ox_entry = os.environ.get("OXYLABS_ENTRYPOINT", "").strip()
    ox_user = _build_oxylabs_username("mantis-resolve")  # placeholder session
    ox_complete = bool(ox_pass and ox_entry and ox_user)

    forced = os.environ.get("MANTIS_PROXY_PROVIDER", "").strip().lower()
    if forced == "oxylabs" and ox_complete:
        return (ox_user.replace("sessid-mantis-resolve", "sessid-{session}"),
                ox_pass, ox_entry.split("://", 1)[-1])
    if forced == "privateproxy" and pp_complete:
        return (pp_user, pp_pass, pp_entry.split("://", 1)[-1])

    if pp_complete:
        return (pp_user, pp_pass, pp_entry.split("://", 1)[-1])
    if ox_complete:
        return (ox_user.replace("sessid-mantis-resolve", "sessid-{session}"),
                ox_pass, ox_entry.split("://", 1)[-1])
    return None


def _ensure_tinyproxy_running(session: str = "mantis") -> str | None:
    """Start tinyproxy with the resolved upstream and return ``host:port``.

    Chromium's ``--proxy-server=URL`` does not reliably honor embedded
    ``user:pass@`` credentials — it produces ``ERR_NO_SUPPORTED_PROXIES``
    on most Chromium builds. Workaround: tinyproxy listens locally with
    no auth, holds the upstream auth, and forwards CONNECT tunnels
    (with a ``Proxy-Authorization`` header) to the upstream proxy.

    Upstream selection is delegated to :func:`_resolve_upstream_proxy` —
    PrivateProxy preferred over Oxylabs because BoatTrader-style
    Cloudflare configs flag Oxylabs residential pool IPs as bot
    traffic.

    Returns ``"127.0.0.1:8888"`` on success, or ``None`` if no
    upstream is available or tinyproxy fails to come up.
    """
    upstream = _resolve_upstream_proxy()
    if not upstream:
        return None
    username_template, password, upstream_host_port = upstream
    # The username MAY contain a ``{session}`` placeholder when the
    # provider supports sticky sessions (Oxylabs). Substitute now.
    username = username_template.format(session=session) if "{session}" in username_template else username_template

    # Write a fresh config every call so a different ``proxy_session``
    # actually rotates the upstream sessid (tinyproxy has no API to
    # change the upstream creds at runtime).
    config = (
        f"Port {_TINYPROXY_PORT}\n"
        "Listen 127.0.0.1\n"
        "Timeout 300\n"
        "MaxClients 32\n"
        "StartServers 2\n"
        f"LogFile \"{_TINYPROXY_LOG}\"\n"
        f"PidFile \"{_TINYPROXY_PID}\"\n"
        # Connect-level logging gives us the upstream CONNECT request /
        # response codes — needed to diagnose 502s without flooding logs.
        "LogLevel Connect\n"
        # Forward all traffic (including HTTPS via CONNECT) to Oxylabs.
        f"upstream http {username}:{password}@{upstream_host_port}\n"
    )
    with open(_TINYPROXY_CONFIG, "w") as fh:
        fh.write(config)

    # Stop any previous tinyproxy (warm container reuse) before starting.
    try:
        if os.path.exists(_TINYPROXY_PID):
            with open(_TINYPROXY_PID) as fh:
                old_pid = int(fh.read().strip() or "0")
            if old_pid:
                os.kill(old_pid, 15)
                time.sleep(0.3)
    except Exception:
        pass

    subprocess.Popen(
        ["tinyproxy", "-c", _TINYPROXY_CONFIG],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait briefly for tinyproxy to bind the port.
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            import socket  # local import — only needed when proxy is used
            with socket.create_connection(("127.0.0.1", _TINYPROXY_PORT), timeout=0.5):
                return f"127.0.0.1:{_TINYPROXY_PORT}"
        except Exception:
            time.sleep(0.2)
    return None


def _wait_for_cf_challenge_clear(
    env,
    *,
    max_seconds: float | None = None,
    poll_interval: float = 1.0,
    min_seconds: float = 2.0,
) -> float:
    """Poll until the Cloudflare interstitial is gone, capped at
    ``max_seconds`` (default 45s, override via
    ``MANTIS_CF_PREWARM_MAX_SECONDS``).

    Returns seconds actually waited. Uses ``env.cdp_evaluate`` to read
    ``document.title`` + a slice of ``document.body.innerText`` and
    look for known CF/anti-bot markers. ``None`` from CDP means the
    page isn't queryable yet — treated as "keep polling", never as
    "cleared". An always-on ``min_seconds`` floor gives the browser a
    beat to start rendering before we declare success on an empty
    page.
    """
    if max_seconds is None:
        try:
            max_seconds = float(
                os.environ.get("MANTIS_CF_PREWARM_MAX_SECONDS", "45")
            )
        except ValueError:
            max_seconds = 45.0

    js = (
        "(() => {"
        "const t = (document.title || '').toLowerCase();"
        "const body = (document.body && document.body.innerText || '')"
        ".slice(0, 4000).toLowerCase();"
        "const blob = t + ' ' + body;"
        "const markers = ["
        "  'just a moment',"
        "  'performing security verification',"
        "  'checking your browser',"
        "  'verifying you are human',"
        "  'verify you are human',"
        "  'enable javascript and cookies to continue'"
        "];"
        "return markers.some(m => blob.includes(m));"
        "})()"
    )

    start = time.monotonic()
    deadline = start + max_seconds
    if min_seconds > 0:
        time.sleep(min(min_seconds, max_seconds))

    while time.monotonic() < deadline:
        try:
            challenge_present = env.cdp_evaluate(js)
        except Exception:
            challenge_present = None
        if challenge_present is False:
            return time.monotonic() - start
        time.sleep(poll_interval)
    return time.monotonic() - start


# ── The function ─────────────────────────────────────────────────────


@app.function(
    image=runner_image,
    secrets=[modal.Secret.from_name("mantis-plan-runner-secrets")],
    cpu=2.0,
    memory=4096,
    timeout=1800,  # hard 30-min cap per plan
)
def run_plan(
    *,
    plan_text: str | None = None,
    plan_json: dict | None = None,
    brain_endpoint: str,
    brain_extra_headers: dict[str, str] | None = None,
    brain_model: str = "Hcompany/Holo3-35B-A3B",
    start_url: str,
    detail_page_pattern: str = "",
    max_cost_usd: float = 10.0,
    max_time_minutes: int = 30,
    use_proxy: bool = False,
    proxy_session: str = "mantis",
    session_name: str = "mantis_run",
    seed: int | None = 42,
) -> dict:
    """Run :class:`MicroPlanRunner` end-to-end inside this container.

    Either ``plan_text`` (decomposed via Claude) or ``plan_json``
    (pre-decomposed micro-plan dict) must be provided. ``plan_json``
    short-circuits the decompose call — useful when the local CLI
    has already paid that cost or when calling from a cached plan.

    Returns the same result-payload dict the local CLI writes to
    ``result.json``: ``plan_signature``, ``session``, ``step_count``,
    ``successes``, ``failures``, ``elapsed_seconds``, ``final_url``,
    ``costs``, ``steps``.
    """
    # Imports are inside the function so the CLI can ``modal.Function
    # .lookup`` without paying the heavy import cost client-side.
    from mantis_agent.brain_holo3 import Holo3Brain
    from mantis_agent.extraction import ClaudeExtractor
    from mantis_agent.grounding import ClaudeGrounding
    from mantis_agent.gym.micro_runner import MicroPlanRunner
    from mantis_agent.gym.result_payload import pack_step as _pack_step
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv
    from mantis_agent.plan_decomposer import MicroPlan, PlanDecomposer
    from mantis_agent.site_config import SiteConfig

    if not plan_text and not plan_json:
        raise ValueError("run_plan: pass either plan_text or plan_json")

    # 1. Resolve / decompose plan.
    if plan_json is not None:
        plan = MicroPlan.from_dict(plan_json)
    else:
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not anthropic_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY missing from Modal Secret — "
                "decomposing plan_text requires Claude access"
            )
        plan = PlanDecomposer().decompose_text(plan_text)

    if not plan.steps:
        raise RuntimeError("decomposed plan has no steps")

    # 2. Bring up Xvfb + (optionally) tinyproxy, then build env.
    # XdotoolGymEnv spawns Chromium itself on the display we
    # provisioned. Chromium's ``--proxy-server`` flag does NOT
    # reliably honor embedded ``user:pass@`` auth — to use the
    # Oxylabs proxy we run tinyproxy locally as the auth holder
    # and point Chromium at ``127.0.0.1:8888`` (no auth). When
    # ``use_proxy=False`` (or Oxylabs Secret is missing), Chromium
    # connects directly via Modal egress.
    _ensure_xvfb_running()
    proxy_host_port = _ensure_tinyproxy_running(proxy_session) if use_proxy else None
    env = XdotoolGymEnv(
        start_url=start_url,
        viewport=_XVFB_VIEWPORT,
        browser="chromium",
        display=_XVFB_DISPLAY,
        proxy_server=proxy_host_port or "",
    )

    # 3. Brain + Claude extractor / grounding.
    brain = Holo3Brain(
        base_url=brain_endpoint,
        model=brain_model,
        extra_headers=brain_extra_headers or None,
    )
    grounding = ClaudeGrounding()
    extractor = ClaudeExtractor()

    # 4. Site config — neutral by default; per-plan tightening via
    # ``--detail-page-pattern`` flows through here.
    site_config = SiteConfig(detail_page_pattern=detail_page_pattern or "")

    # 5. Pre-warm env at start_url before runner.run, mirroring the
    # local CLI's prewarm contract from PR #218. Then poll the page
    # until any Cloudflare JS challenge resolves — replaces the old
    # hardcoded 15s sleep, which was insufficient when CF takes
    # 20-30s on residential proxies and wasteful when it clears in
    # 2-3s. Without this, CF-protected sites show "Performing
    # security verification" on the first screenshot and the gate
    # step rejects with ``gate:FAIL:Error 403``.
    env.reset(task="modal_plan_run", start_url=start_url)
    _wait_for_cf_challenge_clear(env)

    # 6. Construct + run runner.
    runner = MicroPlanRunner(
        brain=brain,
        env=env,
        grounding=grounding,
        extractor=extractor,
        site_config=site_config,
        run_key=session_name,
        session_name=session_name,
        max_cost=max_cost_usd,
        max_time_minutes=max_time_minutes,
        seed=seed,
    )
    t0 = time.time()
    step_results = runner.run(plan, resume=False)
    elapsed = time.time() - t0

    # 7. Build the same result-payload shape the local CLI writes.
    successes = sum(1 for r in step_results if r.success)
    failures = len(step_results) - successes
    final_url = getattr(runner, "_last_known_url", "")
    time_meter = getattr(runner, "time_meter", None)
    wall_time_breakdown = (
        {k: round(v, 3) for k, v in time_meter.breakdown().items()}
        if time_meter is not None else {}
    )
    # Epic #377 Phase C: self-healing audit log.
    from mantis_agent.gym import healing_events as _healing
    healing_log = _healing.snapshot(runner)

    return {
        "plan_signature": runner.plan_signature,
        "session": session_name,
        "step_count": len(step_results),
        "successes": successes,
        "failures": failures,
        # ``total_time_s`` matches ``build_micro_result``'s schema (used
        # by the HTTP API). ``elapsed_seconds`` is the legacy float form
        # this path has always returned — kept as an alias so existing
        # consumers don't break. Both round-trip the same wall-clock.
        "total_time_s": round(elapsed),
        "elapsed_seconds": round(elapsed, 2),
        "wall_time_breakdown": wall_time_breakdown,
        "healing_events": healing_log,
        "final_url": final_url,
        "costs": dict(getattr(runner, "costs", {})),
        "steps": [
            _pack_step(
                r,
                time_breakdown=(
                    time_meter.step_breakdown(r.step_index)
                    if time_meter is not None else None
                ),
            )
            for r in step_results
        ],
    }


# ── Diagnostic: probe the proxy stack from inside the container ─────


@app.function(
    image=runner_image,
    secrets=[modal.Secret.from_name("mantis-plan-runner-secrets")],
    cpu=1.0,
    memory=2048,
    timeout=120,
)
def diagnose_proxy(session: str = "diag") -> dict:
    """Boot tinyproxy + curl through it to validate the upstream auth.

    Returns a structured diagnostic dict with:
    - ``has_credentials``: bool — Modal Secret has Oxylabs vars
    - ``tinyproxy_started``: bool — local listener bound :8888
    - ``tinyproxy_log``: str — last 2KB of tinyproxy log
    - ``ipify_via_proxy``: str — body of ``GET https://api.ipify.org?format=text``
      routed through tinyproxy (or the curl error)
    - ``ipify_via_oxylabs_direct``: str — same URL via curl --proxy http://user:pass@host:port
      (skips tinyproxy entirely; tests Oxylabs auth in isolation)
    - ``ipify_direct``: str — body of the same URL on direct egress
    - ``upstream_username_template``: str — what we'd send to Oxylabs (no pwd)
    """
    out: dict = {}
    out["has_credentials"] = all(
        os.environ.get(k, "").strip()
        for k in ("OXYLABS_USERNAME", "OXYLABS_PASSWORD", "OXYLABS_ENTRYPOINT")
    )
    username = _build_oxylabs_username(session)
    out["upstream_username_template"] = username

    # Direct egress IP first — confirms basic connectivity.
    try:
        proc = subprocess.run(
            ["curl", "-sS", "-m", "10", "https://api.ipify.org?format=text"],
            capture_output=True, timeout=15,
        )
        out["ipify_direct"] = proc.stdout.decode().strip() or proc.stderr.decode()
    except Exception as exc:
        out["ipify_direct"] = f"ERROR: {exc}"

    # Test Oxylabs auth directly via curl (skips tinyproxy). We try both
    # the geo-pinned username and a bare ``customer-USER`` form — if the
    # bare form works but the pinned one doesn't, the Oxylabs plan
    # doesn't have city-level pinning rights. We also URL-escape the
    # password since ``+`` is a special char in URL auth fields.
    import urllib.parse
    entrypoint = os.environ.get("OXYLABS_ENTRYPOINT", "").strip()
    password = os.environ.get("OXYLABS_PASSWORD", "").strip()
    if entrypoint and password and username:
        host_port = entrypoint.split("://", 1)[-1]
        bare_user = f"customer-{os.environ.get('OXYLABS_USERNAME', '').strip()}"
        pwd_quoted = urllib.parse.quote(password, safe="")
        for label, user in (("pinned", username), ("bare", bare_user)):
            try:
                proc = subprocess.run(
                    [
                        "curl", "-sS", "-m", "15", "-w", "\nHTTP %{http_code}",
                        "--proxy", f"http://{user}:{pwd_quoted}@{host_port}",
                        "--proxy-anyauth",
                        "https://api.ipify.org?format=text",
                    ],
                    capture_output=True, timeout=20,
                )
                stdout = proc.stdout.decode().strip()
                stderr = proc.stderr.decode().strip()
                out[f"ipify_via_oxylabs_{label}"] = (
                    stdout or f"STDERR: {stderr} (rc={proc.returncode})"
                )
            except Exception as exc:
                out[f"ipify_via_oxylabs_{label}"] = f"ERROR: {exc}"

    # Also try the dedicated city-level entrypoint (Oxylabs ports 7777
    # for city-pinning vs 10000 for country-only).
    city_entry = os.environ.get("OXYLABS_CITY_ENTRYPOINT", "").strip()
    if city_entry and password and username:
        try:
            city_host_port = city_entry.split("://", 1)[-1]
            pwd_quoted = urllib.parse.quote(password, safe="")
            proc = subprocess.run(
                [
                    "curl", "-sS", "-m", "15", "-w", "\nHTTP %{http_code}",
                    "--proxy", f"http://{username}:{pwd_quoted}@{city_host_port}",
                    "--proxy-anyauth",
                    "https://api.ipify.org?format=text",
                ],
                capture_output=True, timeout=20,
            )
            stdout = proc.stdout.decode().strip()
            stderr = proc.stderr.decode().strip()
            out["ipify_via_oxylabs_city_entry"] = (
                stdout or f"STDERR: {stderr} (rc={proc.returncode})"
            )
        except Exception as exc:
            out["ipify_via_oxylabs_city_entry"] = f"ERROR: {exc}"

    # Also try PrivateProxy as the alternate provider — user keeps both
    # in .env, so when Oxylabs is down we want clear evidence.
    pp_user = os.environ.get("PRIVATEPROXY_USERNAME", "").strip()
    pp_pass = os.environ.get("PRIVATEPROXY_PASSWORD", "").strip()
    pp_entry = os.environ.get("PRIVATEPROXY_ENTRYPOINT", "").strip()
    if pp_user and pp_pass and pp_entry:
        try:
            pp_host_port = pp_entry.split("://", 1)[-1]
            pp_pass_q = urllib.parse.quote(pp_pass, safe="")
            proc = subprocess.run(
                [
                    "curl", "-sS", "-m", "15", "-w", "\nHTTP %{http_code}",
                    "--proxy", f"http://{pp_user}:{pp_pass_q}@{pp_host_port}",
                    "--proxy-anyauth",
                    "https://api.ipify.org?format=text",
                ],
                capture_output=True, timeout=20,
            )
            stdout = proc.stdout.decode().strip()
            stderr = proc.stderr.decode().strip()
            out["ipify_via_privateproxy"] = (
                stdout or f"STDERR: {stderr} (rc={proc.returncode})"
            )
        except Exception as exc:
            out["ipify_via_privateproxy"] = f"ERROR: {exc}"
    else:
        out["ipify_via_privateproxy"] = "no privateproxy creds"

    proxy_host_port = _ensure_tinyproxy_running(session)
    out["tinyproxy_started"] = bool(proxy_host_port)
    out["tinyproxy_host_port"] = proxy_host_port or ""

    # Give tinyproxy a moment to be fully ready, then probe.
    if proxy_host_port:
        try:
            proc = subprocess.run(
                [
                    "curl", "-sS", "-m", "20",
                    "--proxy", f"http://{proxy_host_port}",
                    "https://api.ipify.org?format=text",
                ],
                capture_output=True, timeout=25,
            )
            stdout = proc.stdout.decode().strip()
            stderr = proc.stderr.decode().strip()
            out["ipify_via_proxy"] = stdout or f"STDERR: {stderr} (rc={proc.returncode})"
        except Exception as exc:
            out["ipify_via_proxy"] = f"ERROR: {exc}"
    else:
        out["ipify_via_proxy"] = "tinyproxy not started"

    # Read tinyproxy log tail.
    try:
        with open(_TINYPROXY_LOG) as fh:
            content = fh.read()
        out["tinyproxy_log"] = content[-2048:]
    except Exception as exc:
        out["tinyproxy_log"] = f"<unreadable: {exc}>"

    return out


# ── Entrypoint for ``modal run`` (ad-hoc invocation, no deploy needed) ─


@app.local_entrypoint()
def cli(
    plan_path: str,
    endpoint: str,
    header: str = "",
    start_url: str = "",
    detail_page_pattern: str = "",
    use_proxy: bool = False,
    proxy_session: str = "mantis",
    output_dir: str = "outputs/modal-run",
):
    """Ad-hoc: ``uv run modal run deploy/modal/modal_plan_runner.py --plan-path …``.

    For repeated runs, prefer ``mantis plan run-modal`` which uses the
    deployed function directly (no per-call image build).
    """
    plan_text: str | None = None
    plan_json: dict | None = None
    if plan_path.endswith(".json"):
        with open(plan_path) as fh:
            plan_json = json.load(fh)
    else:
        with open(plan_path) as fh:
            plan_text = fh.read()

    headers: dict[str, str] = {}
    if header:
        for entry in header.split(","):
            k, _, v = entry.partition("=")
            if k and v:
                headers[k.strip()] = v.strip()

    result = run_plan.remote(
        plan_text=plan_text,
        plan_json=plan_json,
        brain_endpoint=endpoint,
        brain_extra_headers=headers or None,
        start_url=start_url,
        detail_page_pattern=detail_page_pattern,
        use_proxy=use_proxy,
        proxy_session=proxy_session,
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "result.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
