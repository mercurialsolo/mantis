## Summary

<!-- 1-3 bullets on what this PR changes and why. -->

## Linked issue

<!-- "Closes #123" or "Refs #123". Delete this section if there isn't one. -->

## Type

- [ ] Bug fix
- [ ] New feature / recipe
- [ ] Refactor (no behavior change)
- [ ] Docs only
- [ ] Build / CI / chore

## Testing

<!-- What did you run? What did it show? -->

- [ ] `pytest tests/ -q`
- [ ] `ruff check .`
- [ ] `mkdocs build --strict` (only if docs changed)
- [ ] Tested against a real browser run (only if runner / env code changed)

## Checklist

- [ ] No customer- or vertical-specific code added to the core; verticals go under `recipes/`
- [ ] No new heavyweight dep added to the base install (`pyproject.toml` extras only)
- [ ] No secrets, API keys, or hostnames committed
- [ ] PR description explains *why*, not *what*
