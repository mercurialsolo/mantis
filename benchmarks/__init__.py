"""Benchmark entry points.

Each module in this package defines its own ``modal.App`` so different
benchmarks can run concurrently on Modal without sharing state. They all
delegate to the shared ``run_osworld_impl`` function in
``deploy/modal/modal_osworld_direct.py`` for the actual agent loop.

Run with:
    uv run modal run --detach benchmarks/osworld_chrome.py
    uv run modal run --detach benchmarks/osworld_multiapp.py
"""
