#!/usr/bin/env python3
"""Convert VisualWebArena task configs to gym-anything environment format.

Reads the raw VWA JSON configs (from VWA_configs/config_files/) and produces
gym-anything-compatible environment directories.

Usage:
    python scripts/convert_vwa_to_gym.py \
        --input VWA_configs/config_files/vwa/test_classifieds.raw.json \
        --output environments/vwa_classifieds \
        --site-url http://localhost:9980

    # Convert all VWA sites
    python scripts/convert_vwa_to_gym.py --all \
        --classifieds-url http://localhost:9980 \
        --shopping-url http://localhost:7770 \
        --reddit-url http://localhost:9999 \
        --wikipedia-url http://localhost:8888

Output structure (per environment):
    environments/vwa_classifieds/
    ├── env.json               # Environment metadata
    ├── tasks.json             # All tasks in flat manifest format
    └── tasks/
        ├── task_0/
        │   ├── task.json      # Task config (intent, eval, etc.)
        │   ├── setup_task.sh  # Navigate to start URL + inject cookies
        │   └── verifier.py    # Evaluation logic
        ├── task_1/
        └── ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# VWA site URL placeholders → env var names
SITE_PLACEHOLDERS = {
    "__CLASSIFIEDS__": "classifieds",
    "__SHOPPING__": "shopping",
    "__REDDIT__": "reddit",
    "__WIKIPEDIA__": "wikipedia",
    "__HOMEPAGE__": "homepage",
    "__SHOPPING_ADMIN__": "shopping_admin",
    "__GITLAB__": "gitlab",
    "__MAP__": "map",
}


def resolve_url(url_template: str | None, site_urls: dict[str, str]) -> str:
    """Replace VWA URL placeholders with actual URLs."""
    if not url_template:
        return ""
    result = url_template
    for placeholder, site_key in SITE_PLACEHOLDERS.items():
        if placeholder in result and site_key in site_urls:
            result = result.replace(placeholder, site_urls[site_key])
    return result


def generate_setup_script(task_config: dict, site_urls: dict[str, str]) -> str:
    """Generate a setup_task.sh for a gym-anything task."""
    start_url = resolve_url(task_config.get("start_url", ""), site_urls)
    storage_state = task_config.get("storage_state", "")

    lines = [
        "#!/usr/bin/env bash",
        "# Auto-generated from VWA task config",
        "set -euo pipefail",
        "",
        f'START_URL="{start_url}"',
        "",
        "# Open the start URL in the browser",
        'if command -v firefox &>/dev/null; then',
        '    firefox "$START_URL" &',
        'elif command -v google-chrome &>/dev/null; then',
        '    google-chrome "$START_URL" &',
        'fi',
        "",
        "# Wait for browser to load",
        "sleep 3",
    ]

    if storage_state:
        lines.extend([
            "",
            "# Note: Authentication cookies should be injected via",
            f"# Playwright storage_state: {storage_state}",
            "# The gym-anything env setup handles this via pre_start hooks.",
        ])

    return "\n".join(lines) + "\n"


def generate_verifier(task_config: dict, site_urls: dict[str, str]) -> str:
    """Generate a verifier.py for a gym-anything task."""
    eval_config = task_config.get("eval", {})
    eval_types = eval_config.get("eval_types", [])
    reference_url = resolve_url(eval_config.get("reference_url", ""), site_urls)
    reference_answers = eval_config.get("reference_answers")
    url_note = eval_config.get("url_note", "EXACT")

    lines = [
        '"""Auto-generated verifier from VWA task config."""',
        "",
        "from __future__ import annotations",
        "",
        "import re",
        "from typing import Any",
        "",
        "",
        "def verify(env, task_config: dict, artifacts: dict[str, Any]) -> dict:",
        '    """Verify task completion.',
        "",
        "    Args:",
        "        env: The gym-anything environment instance.",
        "        task_config: The task configuration dict.",
        "        artifacts: Extracted artifacts from the environment.",
        "",
        "    Returns:",
        '        Dict with "score" (0.0-1.0) and "reason".',
        '    """',
        "    score = 0.0",
        '    reasons = []',
        "",
    ]

    if "url_match" in eval_types:
        lines.extend([
            f'    reference_url = "{reference_url}"',
            f'    url_note = "{url_note}"',
            '    current_url = artifacts.get("current_url", "")',
            "",
            '    if url_note == "EXACT":',
            "        if current_url == reference_url:",
            "            score = max(score, 1.0)",
            '            reasons.append("URL exact match")',
            "        else:",
            '            reasons.append(f"URL mismatch: {current_url} != {reference_url}")',
            '    elif url_note == "GOLD in PRED":',
            "        if reference_url in current_url:",
            "            score = max(score, 1.0)",
            '            reasons.append("URL contains reference")',
            "        else:",
            '            reasons.append(f"URL does not contain {reference_url}")',
            "",
        ])

    if "string_match" in eval_types and reference_answers:
        lines.extend([
            f"    reference_answers = {reference_answers!r}",
            '    agent_answer = artifacts.get("agent_answer", "")',
            "    for ref in reference_answers:",
            "        if ref.lower() in agent_answer.lower():",
            "            score = max(score, 1.0)",
            '            reasons.append(f"String match: {ref}")',
            "            break",
            "    else:",
            '        reasons.append("No string match found")',
            "",
        ])

    if "program_html" in eval_types:
        lines.extend([
            '    # program_html evaluation requires page content extraction',
            '    # This is handled by the gym-anything verification framework',
            '    html_programs = task_config.get("eval", {}).get("program_html", [])',
            "    if html_programs:",
            '        reasons.append("program_html check delegated to framework")',
            "",
        ])

    lines.extend([
        '    return {"score": score, "reason": "; ".join(reasons) if reasons else "No checks ran"}',
        "",
    ])

    return "\n".join(lines) + "\n"


def convert_vwa_task(
    task_config: dict,
    output_dir: Path,
    site_urls: dict[str, str],
) -> dict:
    """Convert a single VWA task to gym-anything format."""
    task_id = f"task_{task_config['task_id']}"
    task_dir = output_dir / "tasks" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # Write task.json
    gym_task = {
        "task_id": task_id,
        "intent": task_config.get("intent", ""),
        "intent_template": task_config.get("intent_template", ""),
        "sites": task_config.get("sites", []),
        "start_url": resolve_url(task_config.get("start_url", ""), site_urls),
        "require_login": task_config.get("require_login", False),
        "require_reset": task_config.get("require_reset", False),
        "image": task_config.get("image"),
        "difficulty": {
            "reasoning": task_config.get("reasoning_difficulty", ""),
            "visual": task_config.get("visual_difficulty", ""),
            "overall": task_config.get("overall_difficulty", ""),
        },
        "eval": task_config.get("eval", {}),
        "timeout": 300,
    }

    with open(task_dir / "task.json", "w") as f:
        json.dump(gym_task, f, indent=2)

    # Write setup_task.sh
    setup_script = generate_setup_script(task_config, site_urls)
    setup_path = task_dir / "setup_task.sh"
    with open(setup_path, "w") as f:
        f.write(setup_script)
    os.chmod(setup_path, 0o755)

    # Write verifier.py
    verifier = generate_verifier(task_config, site_urls)
    with open(task_dir / "verifier.py", "w") as f:
        f.write(verifier)

    return gym_task


def convert_vwa_file(
    input_path: Path,
    output_dir: Path,
    site_urls: dict[str, str],
    site_name: str,
) -> None:
    """Convert an entire VWA config file to a gym-anything environment."""
    with open(input_path) as f:
        tasks = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write env.json
    env_config = {
        "name": f"vwa_{site_name}",
        "description": f"VisualWebArena {site_name} environment",
        "source": "visualwebarena",
        "platform": "linux",
        "resolution": [1920, 1080],
        "browser": "firefox",
        "runner": "docker",
        "total_tasks": len(tasks),
    }
    with open(output_dir / "env.json", "w") as f:
        json.dump(env_config, f, indent=2)

    # Convert each task
    all_tasks = []
    for task_config in tasks:
        gym_task = convert_vwa_task(task_config, output_dir, site_urls)
        all_tasks.append(gym_task)

    # Write flat manifest
    with open(output_dir / "tasks.json", "w") as f:
        json.dump(all_tasks, f, indent=2)

    print(f"Converted {len(tasks)} tasks → {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert VWA task configs to gym-anything environment format"
    )

    parser.add_argument("--input", help="Path to a single VWA .raw.json config file")
    parser.add_argument("--output", help="Output directory for the gym-anything environment")
    parser.add_argument("--site-url", help="Base URL for the site (for single-file mode)")
    parser.add_argument("--site-name", help="Site name (auto-detected from filename if omitted)")

    parser.add_argument("--all", action="store_true", help="Convert all VWA sites")
    parser.add_argument("--vwa-config-dir", default="VWA_configs/config_files/vwa",
                        help="Directory containing VWA raw JSON configs")
    parser.add_argument("--output-base", default="environments",
                        help="Base output directory for all environments")

    # Site URLs (used with --all)
    parser.add_argument("--classifieds-url", default="http://localhost:9980")
    parser.add_argument("--shopping-url", default="http://localhost:7770")
    parser.add_argument("--reddit-url", default="http://localhost:9999")
    parser.add_argument("--wikipedia-url", default="http://localhost:8888")
    parser.add_argument("--homepage-url", default="http://localhost:4399")

    args = parser.parse_args()

    site_urls = {
        "classifieds": args.classifieds_url,
        "shopping": args.shopping_url,
        "reddit": args.reddit_url,
        "wikipedia": args.wikipedia_url,
        "homepage": args.homepage_url,
    }

    if args.all:
        vwa_dir = Path(args.vwa_config_dir)
        output_base = Path(args.output_base)

        for raw_json in sorted(vwa_dir.glob("*.raw.json")):
            # Extract site name: test_classifieds.raw.json → classifieds
            site_name = raw_json.stem.replace(".raw", "").replace("test_", "")
            output_dir = output_base / f"vwa_{site_name}"
            convert_vwa_file(raw_json, output_dir, site_urls, site_name)

    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: {input_path} not found", file=sys.stderr)
            sys.exit(1)

        site_name = args.site_name or input_path.stem.replace(".raw", "").replace("test_", "")

        if args.site_url:
            site_urls[site_name] = args.site_url

        output_dir = Path(args.output) if args.output else Path(f"environments/vwa_{site_name}")
        convert_vwa_file(input_path, output_dir, site_urls, site_name)

    else:
        parser.error("Either --input or --all is required")


if __name__ == "__main__":
    main()
