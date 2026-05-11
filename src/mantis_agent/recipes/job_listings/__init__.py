"""job_listings recipe — public-jobs-board extraction.

Walks a public jobs board (Greenhouse / Lever / Workday-style), opens
each listing in turn, and extracts a structured row per role (title,
team, location, url). No schema.py — the extraction shape lives inline
in the plan's claude step.
"""
