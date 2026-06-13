# exchange-connectivity-hub

RAG system for Asian exchange connectivity documentation

## Setup

```bash
make dev-install                     # install with dev dependencies
git config core.hooksPath .githooks  # enable the pre-push protection
pre-commit install                   # enable pre-commit hooks
```

## Development

```bash
make test       # run tests with coverage
make lint       # ruff check + format check + mypy
make format     # auto-format with ruff
make ci         # full local pipeline (lint + test)
```

## Layout

- `src/`   — application code (import as `from src...`)
- `tests/` — `unit/` and `integration/` suites, shared fixtures in `conftest.py`
