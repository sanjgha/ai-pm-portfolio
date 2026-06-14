# ── Root Makefile ──────────────────────────────────────────────────────────────
# Orchestrates root-level tooling and delegates to individual projects.
# Run `make help` to see all targets.

PROJECTS_DIR := projects
PROJECTS     := $(wildcard $(PROJECTS_DIR)/*)

.PHONY: help install dev-install \
        test test-cov lint format typecheck \
        test-all lint-all ci-all \
        new-project clean

# ── Help ───────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  ai-pm-portfolio — monorepo commands"
	@echo ""
	@echo "  Root tooling"
	@echo "    make install        Install shared package"
	@echo "    make dev-install    Install dev toolchain (ruff, mypy, pytest, ...)"
	@echo "    make test           Run shared tests"
	@echo "    make lint           Lint shared + root code"
	@echo "    make format         Auto-format shared + root code"
	@echo "    make typecheck      Type-check shared code"
	@echo ""
	@echo "  Monorepo-wide"
	@echo "    make test-all       Run tests in every project"
	@echo "    make lint-all       Lint every project"
	@echo "    make ci-all         Full CI across every project"
	@echo ""
	@echo "  Scaffolding"
	@echo "    make new-project name=<slug>   Scaffold a new project under projects/"
	@echo ""

# ── Root-level targets ─────────────────────────────────────────────────────────
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=shared --cov-report=html

lint:
	ruff check shared/ tests/
	ruff format --check shared/ tests/

format:
	ruff format shared/ tests/

typecheck:
	mypy shared/ --ignore-missing-imports

ci: lint test

# ── Monorepo-wide targets ──────────────────────────────────────────────────────
test-all:
	@for p in $(PROJECTS); do \
	  if [ -f "$$p/Makefile" ]; then \
	    echo "\n── $$p ──"; \
	    $(MAKE) -C $$p test; \
	  fi; \
	done

lint-all:
	@for p in $(PROJECTS); do \
	  if [ -f "$$p/Makefile" ]; then \
	    echo "\n── $$p ──"; \
	    $(MAKE) -C $$p lint; \
	  fi; \
	done

ci-all: lint test
	@for p in $(PROJECTS); do \
	  if [ -f "$$p/Makefile" ]; then \
	    echo "\n── $$p ──"; \
	    $(MAKE) -C $$p ci; \
	  fi; \
	done

# ── Scaffolding ────────────────────────────────────────────────────────────────
new-project:
	@test -n "$(name)" || (echo "Usage: make new-project name=<slug>"; exit 1)
	python3 ~/.claude/skills/new-python-project/scripts/scaffold.py \
	  --name "$(name)" \
	  --description "$(desc)" \
	  --python "3.10" \
	  --path "$(PROJECTS_DIR)/$(name)"
	@echo "\nProject scaffolded at $(PROJECTS_DIR)/$(name)"
	@echo "cd $(PROJECTS_DIR)/$(name) && pip install -e '.[dev]' && pre-commit install"

# ── Clean ──────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache build/ dist/ *.egg-info htmlcov/ .coverage
