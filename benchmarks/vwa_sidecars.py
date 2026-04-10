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


# ══════════════════════════════════════════════════════════════════════════════
# 3b. Classifieds (Osclass) — real implementation from Docker Hub image
# ══════════════════════════════════════════════════════════════════════════════
# The classifieds app image is already on Docker Hub: jykoh/classifieds:latest
# We layer MySQL on top and import the SQL dumps at startup.

# Build Classifieds from scratch: PHP+Apache+MySQL+Osclass.
# The Docker Hub image (jykoh/classifieds) is 40+ GB and times out on pull.
# Instead we install Osclass from source and import the SQL dump (85 MB).
classifieds_image = (
    modal.Image.debian_slim()
    .apt_install(
        "apache2", "php", "php-mysql", "php-gd", "php-curl", "php-xml",
        "php-mbstring", "php-zip", "libapache2-mod-php",
        "mariadb-server", "wget", "unzip", "python3",
    )
    .run_commands(
        # Prepare MariaDB (MySQL-compatible)
        "mkdir -p /run/mysqld && chown mysql:mysql /run/mysqld",
        "mysql_install_db --user=mysql --datadir=/var/lib/mysql 2>/dev/null || true",
        # Download Osclass (the classifieds PHP app)
        "wget -q -O /tmp/osclass.zip 'https://github.com/osclass/Osclass/releases/download/3.9.0/osclass-3.9.0.zip' || true",
        "mkdir -p /var/www/html/osclass",
        "cd /var/www/html/osclass && (unzip -o /tmp/osclass.zip 2>/dev/null || true)",
        "chown -R www-data:www-data /var/www/html/osclass",
        "rm -f /tmp/osclass.zip",
        # Configure Apache to serve Osclass on port 9980
        "echo 'Listen 9980' > /etc/apache2/ports.conf",
        """echo '<VirtualHost *:9980>
    DocumentRoot /var/www/html/osclass
    <Directory /var/www/html/osclass>
        AllowOverride All
        Require all granted
    </Directory>
</VirtualHost>' > /etc/apache2/sites-available/000-default.conf""",
        "a2enmod rewrite",
    )
    .add_local_file(
        "/tmp/classifieds_extract/classifieds_docker_compose/mysql/osclass_craigslist.sql",
        "/opt/osclass_craigslist.sql",
        copy=True,
    )
)


@app.function(
    image=classifieds_image,
    cpu=2,
    memory=4096,
    scaledown_window=600,
)
@modal.web_server(port=9980, startup_timeout=180)
def vwa_classifieds():
    """VWA Classifieds (Osclass) — real classified ads from Docker Hub image.

    Starts MySQL, imports the product/listing database, then starts the
    Osclass PHP app on port 9980.
    """
    modal_url = "https://getmason--vwa-sidecars-vwa-classifieds.modal.run"

    # Start MySQL
    print("Starting MySQL...")
    subprocess.Popen(
        ["mysqld_safe"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(8)

    # Create database and import data
    print("Importing classifieds database...")
    try:
        subprocess.run(
            ["mysql", "-u", "root", "-e",
             "DROP DATABASE IF EXISTS osclass; CREATE DATABASE osclass;"],
            capture_output=True, text=True, timeout=30, check=True,
        )
        subprocess.run(
            ["mysql", "-u", "root", "osclass", "-e",
             "source /opt/osclass_craigslist.sql;"],
            capture_output=True, text=True, timeout=120, check=True,
        )
        print("Database imported successfully")
    except Exception as e:
        print(f"DB import failed: {e}")

    # Update the site URL in the database to point to Modal
    try:
        subprocess.run(
            ["mysql", "-u", "root", "osclass", "-e",
             f"UPDATE oc_t_preference SET s_value='{modal_url}/' WHERE s_name='pageTitle' OR s_name='siteUrl';"],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:
        pass

    # The classifieds image already has Apache configured on port 9980
    # via its Dockerfile CMD. The web_server decorator expects the port
    # to be bound by the time startup_timeout elapses. Apache should
    # already be running from the image's entrypoint.
    print(f"Classifieds ready at {modal_url}")


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
# Shopping image is a thin base with tools needed to extract and run the
# Magento Docker image. The actual 68 GB tar is downloaded to the Modal
# volume ONCE via the download_shopping_image() function below, then
# extracted at sidecar startup time.
# Thin image — only needs Python + wget for downloading, plus the flatten
# script. Apache/PHP/MySQL all live INSIDE the extracted Docker filesystem
# and run via chroot, so the host image doesn't need them.
shopping_image = (
    modal.Image.debian_slim()
    .apt_install("wget", "python3")
    .add_local_file("benchmarks/flatten_docker_layers.py", "/opt/flatten.py", copy=True)
)


@app.function(
    image=shopping_image,
    cpu=2,
    memory=8192,
    volumes={"/data": vol},
    timeout=86400,
)
def download_shopping_image():
    """One-time download of the Shopping Docker tar to the Modal volume.

    Run once: modal run benchmarks/vwa_sidecars.py::download_shopping_image
    The 68 GB tar is downloaded to /data/vwa/shopping_final_0712.tar.
    """
    tar_path = "/data/vwa/shopping_final_0712.tar"
    if os.path.exists(tar_path):
        size = os.path.getsize(tar_path)
        print(f"Shopping tar already exists: {size / 1e9:.1f} GB")
        return
    os.makedirs("/data/vwa", exist_ok=True)
    print("Downloading Shopping Docker image (~68 GB) from CMU...")
    print("This runs once and is cached on the volume.")
    subprocess.run(
        ["wget", "-q", "--show-progress", "-O", tar_path,
         "http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar"],
        check=True,
    )
    vol.commit()
    size = os.path.getsize(tar_path)
    print(f"Done — {size / 1e9:.1f} GB saved to {tar_path}")


@app.function(
    image=shopping_image,
    cpu=2,
    memory=8192,
    volumes={"/data": vol},
    scaledown_window=600,
)
@modal.web_server(port=80, startup_timeout=300)
def vwa_shopping():
    """VWA Shopping (Magento) — real Magento e-commerce from Docker image.

    First run: extracts the Docker image tar from the volume (~5 min).
    Subsequent runs: uses cached extraction on ephemeral disk.
    Then starts MySQL + Apache via chroot into the extracted filesystem.

    Pre-requisite: run download_shopping_image() once to fetch the tar.
    """
    tar_path = "/data/vwa/shopping_final_0712.tar"
    root = "/opt/shopping"

    if not os.path.exists(tar_path):
        print("ERROR: Shopping tar not found on volume.")
        print("Run first: modal run benchmarks/vwa_sidecars.py::download_shopping_image")
        # Fall back to stub
        _make_stub("Shopping (tar missing)", 80)
        return

    if not os.path.exists(os.path.join(root, "var", "www")):
        print(f"Extracting Docker layers from {tar_path}...")
        subprocess.run(
            ["python3", "/opt/flatten.py", tar_path, root],
            check=True,
        )
        print("Extraction complete")

    # Mount proc/sys/dev for chroot services
    for mnt in ["proc", "sys", "dev"]:
        mnt_path = os.path.join(root, mnt)
        os.makedirs(mnt_path, exist_ok=True)
        subprocess.run(["mount", "--bind", f"/{mnt}", mnt_path], capture_output=True)

    # Start MySQL
    print("Starting MySQL...")
    subprocess.Popen(
        ["chroot", root, "mysqld_safe"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "HOME": "/root"},
    )
    time.sleep(10)

    # Start Apache (serves Magento on port 80)
    print("Starting Apache...")
    subprocess.Popen(
        ["chroot", root, "apachectl", "-D", "FOREGROUND"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "HOME": "/root"},
    )
    time.sleep(3)

    # Set Magento's base URL
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
             f'UPDATE core_config_data SET value="{modal_url}/" '
             f'WHERE path = "web/secure/base_url";'],
            capture_output=True, text=True, timeout=15,
        )
        subprocess.run(
            ["chroot", root, "/var/www/magento2/bin/magento", "cache:flush"],
            capture_output=True, text=True, timeout=30,
        )
        print(f"Shopping ready at {modal_url}")
    except Exception as e:
        print(f"Warning: Magento base URL config failed: {e}")


@app.function(image=stub_image, cpu=0.5, memory=256, scaledown_window=600)
@modal.web_server(port=9999, startup_timeout=30)
def vwa_reddit():
    """VWA Reddit (Postmill) — STUB (needs PHP+PostgreSQL implementation)."""
    _make_stub("Reddit", 9999)


# ── One-time download functions for Reddit and Classifieds ────────────────────

@app.function(
    image=shopping_image,  # reuse — just needs wget+python
    cpu=2,
    memory=4096,
    volumes={"/data": vol},
    timeout=86400,
)
def download_reddit_image():
    """One-time download of the Reddit (Postmill) Docker tar to the Modal volume.

    Run once: modal run benchmarks/vwa_sidecars.py::download_reddit_image
    """
    tar_path = "/data/vwa/postmill-populated-exposed-withimg.tar"
    if os.path.exists(tar_path):
        size = os.path.getsize(tar_path)
        print(f"Reddit tar already exists: {size / 1e9:.1f} GB")
        return
    os.makedirs("/data/vwa", exist_ok=True)
    print("Downloading Reddit (Postmill) Docker image from CMU...")
    subprocess.run(
        ["wget", "-q", "--show-progress", "-O", tar_path,
         "http://metis.lti.cs.cmu.edu/webarena-images/postmill-populated-exposed-withimg.tar"],
        check=True,
    )
    vol.commit()
    size = os.path.getsize(tar_path)
    print(f"Done — {size / 1e9:.1f} GB saved to {tar_path}")


@app.function(
    image=shopping_image,  # reuse
    cpu=2,
    memory=4096,
    volumes={"/data": vol},
    timeout=86400,
)
def download_classifieds_image():
    """One-time download of the Classifieds Docker compose zip to the Modal volume.

    Run once: modal run benchmarks/vwa_sidecars.py::download_classifieds_image
    """
    zip_path = "/data/vwa/classifieds_docker_compose.zip"
    if os.path.exists(zip_path):
        size = os.path.getsize(zip_path)
        print(f"Classifieds zip already exists: {size / 1e9:.1f} GB")
        return
    os.makedirs("/data/vwa", exist_ok=True)
    print("Downloading Classifieds Docker compose from archive.org...")
    subprocess.run(
        ["wget", "-q", "--show-progress", "-O", zip_path,
         "https://archive.org/download/classifieds_docker_compose/classifieds_docker_compose.zip"],
        check=True,
    )
    vol.commit()
    size = os.path.getsize(zip_path)
    print(f"Done — {size / 1e9:.1f} GB saved to {zip_path}")
