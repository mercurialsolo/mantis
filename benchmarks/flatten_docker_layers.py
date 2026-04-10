#!/usr/bin/env python3
"""Flatten Docker image layers from a tar archive.

Usage: python3 flatten_docker_layers.py <tar_path> <output_dir>

Reads manifest.json from the tar to get layer order, extracts each
layer on top of the previous one, then cleans up Docker whiteout
files (.wh.*) that mark deletions between layers.
"""

import json
import os
import subprocess
import sys


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <tar_path> <output_dir>")
        sys.exit(1)

    tar_path = sys.argv[1]
    output_dir = sys.argv[2]
    tmp_dir = "/tmp/docker_layers"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # Extract the outer tar (contains manifest + layer tars)
    print(f"Extracting {tar_path}...")
    subprocess.run(["tar", "xf", tar_path, "-C", tmp_dir], check=True)

    # Read manifest to get layer order
    with open(os.path.join(tmp_dir, "manifest.json")) as f:
        manifest = json.load(f)

    layers = manifest[0]["Layers"]
    print(f"Found {len(layers)} layers")

    # Apply each layer in order
    for i, layer in enumerate(layers):
        layer_path = os.path.join(tmp_dir, layer)
        print(f"  Applying layer {i + 1}/{len(layers)}: {layer[:40]}...")
        subprocess.run(
            ["tar", "xf", layer_path, "-C", output_dir],
            capture_output=True,
        )

    # Clean up whiteout files (Docker's layer deletion markers)
    wh_count = 0
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.startswith(".wh."):
                target_name = f[4:]  # Remove .wh. prefix
                target_path = os.path.join(root, target_name)
                wh_path = os.path.join(root, f)
                if os.path.exists(target_path):
                    subprocess.run(["rm", "-rf", target_path], capture_output=True)
                os.remove(wh_path)
                wh_count += 1

    print(f"Cleaned up {wh_count} whiteout files")

    # Clean up temp
    subprocess.run(["rm", "-rf", tmp_dir], capture_output=True)
    print(f"Done — filesystem at {output_dir}")


if __name__ == "__main__":
    main()
