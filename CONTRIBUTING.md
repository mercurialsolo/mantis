# Contributing to Mantis

Thanks for considering a contribution. Mantis is a CUA (computer-use agent)
that drives a real browser via Xvfb + Chrome + xdotool, with Holo3 for tactical
clicks and Claude for surgical reasoning. The goal of this project is to be a
generic, host-agnostic agent that can execute *any* structured plan — not just
the listing-extraction plan it ships with today.

If you're contributing toward that goal (more recipes, more plan primitives,
more pluggable model backends, better verifiers), you're in the right place.

## Ground rules

1. **Open an issue first** for non-trivial changes. A 30-second sketch of the
   approach saves a 30-hour rework loop.
2. **One concern per PR.** Refactors and feature work go in separate PRs.
3. **No customer- or vertical-specific code in the core.** If your change adds
   a domain-specific schema, prompt, or selector, it goes under
   `recipes/<vertical>/` and is wired in via the registry, not hardcoded into
   `extraction.py` or `micro_runner.py`.
4. **Tests are required for new behavior.** Mock the LLM brain (`brain_*.py`)
   so tests run on CI without GPU and without API keys.

## Development setup

```bash
git clone https://github.com/mercurialsolo/mantis
cd mantis
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,server,orchestrator,metrics,docs]"
pre-commit install   # if you add the hook
```

For full-stack work that drives a real browser you also need:

```bash
sudo apt-get install -y ffmpeg xvfb chromium-browser xdotool   # Linux
pip install -e ".[local-cua]"
```

## Running checks locally

```bash
pytest tests/ -q          # unit + orchestrator-surface tests
ruff check .              # lint
mkdocs build --strict     # docs must build clean
```

CI runs all three on every PR (`.github/workflows/test.yml`). PRs that fail
any of them won't merge.

## Commit & PR conventions

- **Commit subject ≤ 72 chars**, imperative mood: `fix:`, `feat:`, `docs:`,
  `refactor:`, `test:`, `chore:`. Follow the existing log style — `git log`
  shows the convention.
- **PR description** must explain *why*, link the issue if any, and list
  what was tested. A diff that explains *what* doesn't help reviewers — the
  diff already shows that.
- **No `--no-verify`** on commits or merges. If a hook fails, fix the cause.
- **Don't amend / force-push** a PR branch once review has started — push
  fixup commits instead.

## Adding a new recipe

A recipe is a self-contained extraction or workflow pattern (jobs board,
e-commerce listing, news article, real-estate listing, etc.). The contract:

1. Plan JSON under `recipes/<name>/plan.json`.
2. Optional `ExtractionSchema` Python under `recipes/<name>/schema.py`.
3. Optional verifier under `recipes/<name>/verifier.py`.
4. README under `recipes/<name>/README.md` with:
   - One-paragraph description
   - Sites it has been tested against
   - Known limitations

Recipes must not import customer-specific configuration. If you need to
parametrize a selector or URL pattern, accept it as a plan field, not a
hardcoded constant.

## Adding a new model backend

Implement the `Brain` protocol (`mantis_agent/brain.py`) and register it via
the `MANTIS_BRAIN` env var or the `register_brain()` helper. Don't touch
`brain_holo3.py` / `brain_claude.py` for unrelated behavior changes — they
are reference implementations.

## Reporting security issues

See [SECURITY.md](SECURITY.md). **Do not open public issues for vulnerabilities.**

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating you agree to abide by its terms.

## License

By contributing, you agree your contributions are licensed under the
[Apache License 2.0](LICENSE) — the same terms as the rest of the project.
