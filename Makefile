.PHONY: help install install-dev install-full sync lint fmt test test-serial docs docs-build precommit precommit-install clean

# Use uv when available — it is ~5-10x faster and locks against uv.lock.
# Fall back to pip + python -m venv for environments without uv.
UV := $(shell command -v uv 2>/dev/null)
ifeq ($(UV),)
  PIP_INSTALL := pip install
  RUN := python -m
else
  PIP_INSTALL := uv pip install
  RUN := uv run
endif

help:  ## Show this help (default target)
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install the slim base package (editable)
	$(PIP_INSTALL) -e .

install-dev:  ## Install dev + server + orchestrator + metrics + docs extras
	$(PIP_INSTALL) -e ".[dev,server,orchestrator,metrics,docs]"

install-full:  ## Install everything including local-cua (torch, transformers, ...)
	$(PIP_INSTALL) -e ".[full,dev,docs]"

sync:  ## Reproduce the locked dev environment from uv.lock (requires uv)
	@if [ -z "$(UV)" ]; then echo "uv not installed — run: pipx install uv  (or curl -LsSf https://astral.sh/uv/install.sh | sh)"; exit 1; fi
	uv sync --extra dev --extra server --extra orchestrator --extra metrics --extra docs

lint:  ## Run ruff lint
	$(RUN) ruff check .

fmt:  ## Apply ruff auto-fixes + format
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

test:  ## Run the test suite in parallel (pytest -n auto via pyproject)
	$(RUN) pytest tests/ -q

test-serial:  ## Run tests serially — use when debugging with --pdb
	$(RUN) pytest tests/ -q -p no:xdist

docs:  ## Serve the MkDocs site locally at http://127.0.0.1:8000
	$(RUN) mkdocs serve

docs-build:  ## Build the docs site with --strict (mirrors CI)
	$(RUN) mkdocs build --strict --site-dir _site

precommit-install:  ## Install the git pre-commit hook
	$(RUN) pre-commit install

precommit:  ## Run pre-commit against all tracked files
	$(RUN) pre-commit run --all-files

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache _site site build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
