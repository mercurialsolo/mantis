"""VWA sidecar services as Modal persistent web servers.

Each VisualWebArena task requires up to 5 backend web services running:
  1. Homepage     (Flask, port 4399)  — simple landing page
  2. Wikipedia    (Kiwix, port 8888)  — offline Wikipedia via ZIM file
  3. Classifieds  (PHP+MySQL, port 9980) — classified ads app
  4. Shopping     (Magento+MySQL, port 7770) — e-commerce store
  5. Reddit       (Postmill+PostgreSQL, port 9999) — Reddit clone

Instead of Docker-in-Docker, each service runs as a Modal
``@modal.web_server()`` function — a persistent container with its own
URL that stays warm between benchmark runs. The VWA benchmark connects
to these URLs via the VWA_ENDPOINTS env vars.

Architecture:
    Modal Container (GPU, A100)           Modal Containers (CPU, persistent)
    ┌──────────────────────┐              ┌──────────────────────┐
    │ benchmarks/           │   HTTP       │ vwa-homepage         │
    │   visualwebarena.py  │──────────────│ Flask on :4399       │
    │                      │              └──────────────────────┘
    │ Gemma 4 26B +        │              ┌──────────────────────┐
    │ Playwright +         │   HTTP       │ vwa-wikipedia        │
    │ SoM grounding        │──────────────│ Kiwix on :8888       │
    │                      │              └──────────────────────┘
    └──────────────────────┘              ┌──────────────────────┐
                                          │ vwa-classifieds      │
                                          │ Apache+PHP+MySQL     │
                                          └──────────────────────┘
                                          ... (shopping, reddit)

Deploy:
    uv run modal deploy benchmarks/vwa_sidecars.py

This deploys all sidecar services as always-on Modal web endpoints.
The GPU benchmark function then connects to them by URL.

Status:
  ✅ Homepage (Flask)   — implemented
  ✅ Wikipedia (Kiwix)  — implemented (needs ZIM file on volume)
  🔲 Classifieds        — TODO (PHP+MySQL, complex)
  🔲 Shopping           — TODO (Magento+MySQL, most complex)
  🔲 Reddit             — TODO (Postmill+PostgreSQL, complex)
"""

from __future__ import annotations

import os
import subprocess
import time

import modal

# ── Shared volume for data (ZIM files, DB dumps, etc.) ───────────────────────
vol = modal.Volume.from_name("osworld-data", create_if_missing=True)

app = modal.App("vwa-sidecars")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Homepage — simple Flask app
# ══════════════════════════════════════════════════════════════════════════════

homepage_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("flask")
)


@app.function(
    image=homepage_image,
    cpu=0.5,
    memory=256,
    scaledown_window=600,  # stay warm for 10 min
)
@modal.web_server(port=4399, startup_timeout=30)
def vwa_homepage():
    """VWA Homepage service — a simple landing page with links to other services."""
    import tempfile

    # Write a minimal Flask app that mimics VWA's homepage
    app_code = '''
from flask import Flask, jsonify
app = Flask(__name__)

@app.route("/")
def index():
    return """
    <html><head><title>VWA Homepage</title></head>
    <body>
        <h1>Welcome to WebArena</h1>
        <ul>
            <li><a href="/classifieds">Classifieds</a></li>
            <li><a href="/shopping">Shopping</a></li>
            <li><a href="/reddit">Reddit</a></li>
            <li><a href="/wikipedia">Wikipedia</a></li>
        </ul>
    </body></html>
    """

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4399)
'''
    tmpfile = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    tmpfile.write(app_code)
    tmpfile.close()
    subprocess.Popen(["python", tmpfile.name])


# ══════════════════════════════════════════════════════════════════════════════
# 2. Wikipedia — Kiwix server with ZIM file
# ══════════════════════════════════════════════════════════════════════════════

wikipedia_image = (
    modal.Image.debian_slim()
    .apt_install("kiwix-tools")
)


@app.function(
    image=wikipedia_image,
    cpu=1,
    memory=2048,
    volumes={"/data": vol},
    scaledown_window=600,
)
@modal.web_server(port=8888, startup_timeout=120)
def vwa_wikipedia():
    """VWA Wikipedia service — offline Wikipedia via Kiwix.

    Requires a ZIM file at /data/vwa/wikipedia.zim. Download once:
        wget -O /tmp/wiki.zim https://download.kiwix.org/zim/wikipedia/wikipedia_en_wp1_mini_2024-07.zim
        modal volume put osworld-data /tmp/wiki.zim vwa/wikipedia.zim
    """
    zim_path = "/data/vwa/wikipedia.zim"
    if not os.path.exists(zim_path):
        print(f"ERROR: ZIM file not found at {zim_path}")
        print("Download it first:")
        print("  wget -O /tmp/wiki.zim https://download.kiwix.org/zim/wikipedia/wikipedia_en_wp1_mini_2024-07.zim")
        print("  modal volume put osworld-data /tmp/wiki.zim vwa/wikipedia.zim")
        # Start a dummy server so the container doesn't crash
        subprocess.Popen(["python3", "-m", "http.server", "8888"])
        return

    subprocess.Popen(["kiwix-serve", "--port", "8888", "--address", "0.0.0.0", zim_path])


# ══════════════════════════════════════════════════════════════════════════════
# 3-5. Classifieds, Shopping, Reddit — stubs for now
# ══════════════════════════════════════════════════════════════════════════════
# These require Apache/Nginx + PHP-FPM + MySQL/PostgreSQL + pre-loaded data.
# Building them as Modal services requires replicating the Docker image setup
# (install PHP, MySQL, import SQL dumps, configure Apache vhosts, etc.).
#
# For now, they are STUBS that return a placeholder page. When we're ready
# to build them out:
#   1. Export the Docker image's filesystem as a tar
#   2. Use modal.Image.run_commands() to replicate the install steps
#   3. Use supervisord to manage Apache + MySQL in one container
#   4. Pre-load DB dumps from the Modal volume
#
# Priority: Shopping (Magento) is needed by 466+210 = 676 tasks.
# Classifieds is needed by 234 tasks. Reddit by 210 tasks.


stub_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("flask")
)


def _make_stub(name: str, port: int):
    """Create a stub Flask app for an unimplemented sidecar."""
    import tempfile
    app_code = f'''
from flask import Flask
app = Flask(__name__)

@app.route("/", defaults={{"path": ""}})
@app.route("/<path:path>")
def catch_all(path):
    return f"""
    <html><head><title>{name} (stub)</title></head>
    <body>
        <h1>{name} — Stub Service</h1>
        <p>This VWA sidecar is not yet fully implemented.</p>
        <p>Requested path: /{{path}}</p>
    </body></html>
    """, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port={port})
'''
    tmpfile = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    tmpfile.write(app_code)
    tmpfile.close()
    subprocess.Popen(["python", tmpfile.name])


@app.function(image=stub_image, cpu=0.5, memory=256, scaledown_window=600)
@modal.web_server(port=9980, startup_timeout=30)
def vwa_classifieds():
    """VWA Classifieds — STUB (needs PHP+MySQL implementation)."""
    _make_stub("Classifieds", 9980)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Shopping (Magento) — real implementation from Docker image
# ══════════════════════════════════════════════════════════════════════════════
# Built from the official VWA Docker image: shopping_final_0712.tar
# Pipeline: docker load → docker push to GHCR → modal.Image.from_registry
#
# The image contains Apache + PHP + MySQL + Magento 2 with pre-loaded products.
# A startup script runs MySQL and Apache inside the container.
# After startup, Magento's base-url is set to the Modal web_server URL.

# Build the Shopping image entirely on Modal — download the official VWA
# Docker tar from archive.org, extract (flatten layers), and use as the
# base filesystem. No local Docker or GHCR push needed.
shopping_image = (
    modal.Image.debian_slim()
    .apt_install("wget", "python3", "jq")
    .run_commands(
        # Download the Shopping Docker image tar (~6 GB)
        "wget -q -O /tmp/shopping.tar 'http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar'",
        # Extract and flatten Docker layers into /opt/shopping
        "mkdir -p /opt/shopping /tmp/layers",
        "tar xf /tmp/shopping.tar -C /tmp/layers",
        # Flatten layers in order from manifest.json
        """python3 -c "
import json, subprocess, os
with open('/tmp/layers/manifest.json') as f:
    manifest = json.load(f)
for layer in manifest[0]['Layers']:
    layer_path = os.path.join('/tmp/layers', layer)
    subprocess.run(['tar', 'xf', layer_path, '-C', '/opt/shopping'], capture_output=True)
# Clean up whiteout files
for root, dirs, files in os.walk('/opt/shopping'):
    for f in files:
        if f.startswith('.wh.'):
            target = os.path.join(root, f.replace('.wh.', ''))
            wh = os.path.join(root, f)
            if os.path.exists(target):
                subprocess.run(['rm', '-rf', target])
            os.remove(wh)
print('Layers flattened successfully')
" """,
        # Clean up the tar to save space in the image
        "rm -rf /tmp/shopping.tar /tmp/layers",
    )
)


@app.function(
    image=shopping_image,
    cpu=2,
    memory=4096,
    scaledown_window=600,
)
@modal.web_server(port=80, startup_timeout=120)
def vwa_shopping():
    """VWA Shopping (Magento) — real Magento e-commerce from Docker image.

    The filesystem was extracted from the official VWA Docker image during
    Modal image build. This startup function uses chroot to run MySQL and
    Apache from the extracted filesystem at /opt/shopping.
    """
    root = "/opt/shopping"

    # Start MySQL from the extracted filesystem
    subprocess.Popen(
        ["chroot", root, "mysqld_safe"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "HOME": "/root"},
    )
    time.sleep(8)  # MySQL needs time to initialize

    # Start Apache (serves Magento on port 80)
    subprocess.Popen(
        ["chroot", root, "apachectl", "-D", "FOREGROUND"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "HOME": "/root"},
    )
    time.sleep(3)

    # Set Magento's base URL to the Modal web_server URL
    modal_url = "https://getmason--vwa-sidecars-vwa-shopping.modal.run"
    try:
        subprocess.run(
            ["chroot", root, "/var/www/magento2/bin/magento",
             "setup:store-config:set", f"--base-url={modal_url}/"],
            capture_output=True, text=True, timeout=60,
        )
        subprocess.run(
            ["chroot", root, "mysql", "-u", "magentouser", "-pMyPassword",
             "magentodb", "-e",
             f'UPDATE core_config_data SET value="{modal_url}/" WHERE path = "web/secure/base_url";'],
            capture_output=True, text=True, timeout=15,
        )
        subprocess.run(
            ["chroot", root, "/var/www/magento2/bin/magento", "cache:flush"],
            capture_output=True, text=True, timeout=30,
        )
        print(f"Shopping base URL set to {modal_url}")
    except Exception as e:
        print(f"Warning: could not set Magento base URL: {e}")


@app.function(image=stub_image, cpu=0.5, memory=256, scaledown_window=600)
@modal.web_server(port=9999, startup_timeout=30)
def vwa_reddit():
    """VWA Reddit (Postmill) — STUB (needs PHP+PostgreSQL implementation)."""
    _make_stub("Reddit", 9999)
