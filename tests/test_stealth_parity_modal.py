"""Source-level checks for the stealth-parity-with-reference-browser PR.

A reference parity-browser stack runs its CUA without tripping
Cloudflare Turnstile; mantis was tripping it on the same proxy
IP. This PR closes 5 gaps + 1 proxy-routing bug:

  1. Locale + Timezone explicitly set (LANG=en_US.UTF-8, TZ=America/New_York)
  2. WebGL flags forcing SwiftShader software renderer
  3. Install Liberation/DejaVu/Noto fonts (sparse font set is a bot tell)
  4. WebGL spoof Chrome extension loaded via --load-extension
  5. Persistent CF clearance cookies via persistent user-data-dir (already in place)
  + proxy_provider preserved through the /v1/predict micro JSON path
    (was silently dropped, causing fallback to iproyal stale creds)

Tests below are source-level (assert wiring is present) — full end-to-
end stealth checks require a live Cloudflare-protected site and a
running Modal deploy, so they live in run_boattrader_urlnav_with_proxy
under scripts/ rather than CI.
"""

from __future__ import annotations

import inspect

import pytest

# ``deploy/modal/modal_cua_server.py`` imports the ``modal`` SDK at
# module-load time, so every test below transitively requires it.
# CI doesn't install modal — same skip gate as test_modal_endpoint.py.
pytest.importorskip("modal")


# ── 1. Locale + TZ env on every Chrome-launching Modal image ─────────────


def test_executor_image_has_locale_tz_env():
    """The vLLM executor image (run_evocua, run_opencua, run_fara)
    must export LANG/LC_ALL/TZ so child Chrome reports a US locale +
    America/New_York timezone matching our US proxy IP."""
    src = _read_modal_server()
    # The .env({...}) block right after the executor_image
    # run_commands. Match all three keys appearing inside an .env()
    # somewhere in the file (presence-of-wiring check; the precise
    # block belongs to executor_image).
    assert '"LANG": "en_US.UTF-8"' in src
    assert '"LC_ALL": "en_US.UTF-8"' in src
    assert '"TZ": "America/New_York"' in src


def test_locale_gen_runs_during_image_build():
    """``locale-gen`` must actually generate en_US.UTF-8 — setting
    LANG without the locale data still leaves Chrome falling back
    to POSIX/C, which is a stronger bot tell than UTC."""
    src = _read_modal_server()
    assert "locale-gen" in src
    assert "en_US.UTF-8" in src


def test_tz_symlink_set_during_image_build():
    """``/etc/localtime`` must symlink to America/New_York so
    container-level system calls (not just Chrome's per-process env)
    also resolve to NYC."""
    src = _read_modal_server()
    assert "America/New_York" in src
    assert "/etc/localtime" in src


# ── 2. WebGL flags forcing SwiftShader ───────────────────────────────────


def test_chrome_launch_has_swiftshader_flags():
    """xdotool_env._start_browser must launch Chrome with WebGL
    forced ON via SwiftShader software rendering. On a virtualized
    GPU (Modal containers don't expose the host GPU to Chrome)
    WebGL might be disabled entirely — which fingerprints as
    'WebGL absent' and CF Turnstile flags it as automation."""
    from mantis_agent.gym import xdotool_env
    src = inspect.getsource(xdotool_env)
    assert "--use-gl=angle" in src
    assert "--use-angle=swiftshader-webgl" in src
    assert "--enable-unsafe-swiftshader" in src
    assert "--ignore-gpu-blocklist" in src
    assert "--enable-webgl" in src


def test_chrome_launch_loads_webgl_spoof_extension():
    """The WebGL spoof extension must be loaded via --load-extension
    when the extension directory exists at the canonical Modal path."""
    from mantis_agent.gym import xdotool_env
    src = inspect.getsource(xdotool_env)
    assert "/opt/chrome-extensions/webgl-spoof" in src
    assert "--load-extension=" in src


# ── 3. Font set parity with the reference browser ───────────────────────


def test_image_installs_stealth_fonts():
    """fonts-liberation + fonts-dejavu-core + fonts-noto-color-emoji
    must be in the apt install list. CF/Turnstile fingerprints
    ~30 canary fonts via document.fonts.check; a Liberation-only
    set is exclusively seen on Linux servers."""
    src = _read_modal_server()
    assert "fonts-liberation" in src
    assert "fonts-dejavu-core" in src
    assert "fonts-noto-color-emoji" in src


def test_stealth_apt_constant_used():
    """The constant ``_STEALTH_APT_FONTS_AND_LOCALE`` should be
    referenced in image definitions so the package set stays
    consistent across executor variants."""
    src = _read_modal_server()
    assert "_STEALTH_APT_FONTS_AND_LOCALE" in src


# ── 4. WebGL spoof extension shipped in Modal images ─────────────────────


def test_webgl_spoof_extension_exists_locally():
    """The chrome_extensions/webgl_spoof/ dir must exist with both
    manifest.json + content.js — these are mounted via add_local_dir."""
    import os
    from deploy.modal import modal_cua_server as m
    assert os.path.isdir(m._WEBGL_SPOOF_LOCAL), (
        f"missing: {m._WEBGL_SPOOF_LOCAL}"
    )
    assert os.path.isfile(
        os.path.join(m._WEBGL_SPOOF_LOCAL, "manifest.json"),
    )
    assert os.path.isfile(
        os.path.join(m._WEBGL_SPOOF_LOCAL, "content.js"),
    )


def test_extension_mounted_on_chrome_launching_images():
    """All 4 Chrome-launching Modal images must mount the extension
    dir at /opt/chrome-extensions/webgl-spoof (Modal-side path that
    xdotool_env looks for at launch). The api_image + planner_image
    don't launch Chrome and don't need it."""
    src = _read_modal_server()
    # Each Chrome-launching image should add the extension dir.
    # We don't have 4 separate symbol bindings (run_holo3 +
    # run_gemma4_cua_worker use inline images chained off
    # planner_base_image), so the count of mounts is the load-
    # bearing assertion.
    mount_count = src.count("add_local_dir(_WEBGL_SPOOF_LOCAL")
    assert mount_count >= 4, (
        f"expected ≥4 webgl-spoof mounts (executor_image, "
        f"claude_executor_image, run_holo3 image, run_gemma4_cua_worker "
        f"image), found {mount_count}"
    )


def test_extension_manifest_is_main_world_content_script():
    """The extension must run as a MAIN-world content script at
    document_start — addScriptToEvaluateOnNewDocument runs in
    ISOLATED world by default which CF detection scripts can
    probe around."""
    import json
    from deploy.modal import modal_cua_server as m
    import os
    with open(os.path.join(m._WEBGL_SPOOF_LOCAL, "manifest.json")) as f:
        manifest = json.load(f)
    assert manifest.get("manifest_version") == 3
    scripts = manifest.get("content_scripts", [])
    assert scripts, "missing content_scripts"
    s0 = scripts[0]
    assert s0.get("world") == "MAIN"
    assert s0.get("run_at") == "document_start"
    assert s0.get("all_frames") is True


# ── 5. proxy_provider preservation through /v1/predict micro path ────────


def test_build_suite_from_payload_preserves_proxy_provider():
    """The micro JSON path of /v1/predict used to drop ``proxy_provider``
    when calling build_micro_suite — downstream build_proxy_config
    would then default to ``iproyal`` (stale creds), producing
    runs that egressed via Modal IPs even when the caller asked
    for PrivateProxy. Regression guard."""
    from deploy.modal import modal_cua_server as m
    src = inspect.getsource(m._build_suite_from_payload)
    # Must thread proxy_provider into build_micro_suite.
    assert "proxy_provider=" in src, (
        "proxy_provider must be passed to build_micro_suite from "
        "the micro JSON path"
    )


def test_build_suite_from_payload_stamps_plan_name():
    """#638 axis 2 follow-up: ``_plan_name`` must be stamped on the
    suite so the Augur Runs list can group runs of the same plan.
    Default is the source file stem; ``payload['plan_name']`` wins
    when the caller sets it explicitly."""
    from deploy.modal import modal_cua_server as m
    src = inspect.getsource(m._build_suite_from_payload)
    assert '_plan_name' in src, (
        "_plan_name must be stamped onto the suite from the micro JSON "
        "path so Augur tags get a human-readable plan identifier"
    )
    assert "path.stem" in src, (
        "_plan_name must default to the source file stem when payload "
        "doesn't supply an explicit plan_name override"
    )


def test_run_executor_emits_plan_name_tag_to_augur():
    """#638 axis 2 follow-up: RunExecutor must forward ``runner.plan_name``
    into the AugurAdapter ``extra_tags`` dict so the Augur Runs list
    can group runs of the same plan. Source-level check matches the
    existing plan_signature / plan_step_count pattern."""
    from mantis_agent.gym import run_executor as re_mod
    src = inspect.getsource(re_mod)
    assert '"plan_name":' in src, (
        "RunExecutor must emit a 'plan_name' tag in the AugurAdapter "
        "extra_tags dict"
    )
    assert 'getattr(runner, "plan_name"' in src, (
        "plan_name must be read off the runner via getattr so single-"
        "container runs without the attribute don't AttributeError"
    )


def test_build_micro_suite_stamps_proxy_provider_into_suite():
    """build_micro_suite must persist ``proxy_provider`` into the
    suite dict at ``_proxy_provider`` so downstream executors
    (setup_env, _run_claude_executor, the holo3 task_loop) read
    the right provider when constructing the proxy config."""
    from mantis_agent.server_utils import build_micro_suite
    suite = build_micro_suite(
        [{"name": "navigate", "url": "https://example.com"}],
        domain="example.com",
        proxy_provider="privateproxy",
        proxy_city="miami",
        proxy_state="fl",
    )
    assert suite["_proxy_provider"] == "privateproxy"
    assert suite["_proxy_city"] == "miami"
    assert suite["_proxy_state"] == "fl"


def test_build_proxy_config_default_is_privateproxy():
    """Default provider when MANTIS_PROXY_PROVIDER + caller provider
    are both unset must be ``privateproxy`` (the actively-maintained
    creds in .env), NOT ``iproyal`` (stale, returns None config,
    silent fallback to Modal IP)."""
    from mantis_agent import server_utils
    src = inspect.getsource(server_utils.build_proxy_config)
    # The default in the (provider or env or "...") chain.
    assert '"privateproxy"' in src
    # The old default should be gone from the default chain (the
    # branch handlers themselves still mention iproyal — those
    # are fine; we're checking the fallback default).
    assert 'or "iproyal"' not in src


# ── Helpers ──────────────────────────────────────────────────────────────


def _read_modal_server() -> str:
    """Load modal_cua_server.py source as text. We avoid importing
    the module because importing it pulls in modal SDK side-effects
    (App creation, Volume.from_name)."""
    from deploy.modal import modal_cua_server as m
    with open(m.__file__) as f:
        return f.read()
