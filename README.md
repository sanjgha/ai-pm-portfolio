# AI PM Portfolio

Monorepo showcasing AI Product Manager skills through working applications —
capital markets tools, automation agents, and decision-support systems built with
modern Python and AI APIs.

## Quick Start

```bash
git config core.hooksPath .githooks   # enable branch protection
make dev-install                      # install root dev toolchain
pre-commit install                    # wire up pre-commit hooks
make ci                               # lint + test root code
```

## Adding a Project

```bash
make new-project name=<slug> desc="One-line description"
cd projects/<slug>
pip install -e ".[dev]"
pre-commit install
make ci
```

## Monorepo Commands

| Command | What it does |
|---------|-------------|
| `make ci` | Lint + test root shared code |
| `make ci-all` | CI across every project |
| `make test-all` | Tests across every project |
| `make lint-all` | Lint across every project |
| `make new-project name=X desc=Y` | Scaffold a new project |
| `make help` | Full command reference |

## Portfolio Projects

| Project | Domain | Status |
|---------|--------|--------|
| _(none yet — run `make new-project` to add the first one)_ | | |

## Layout

```
shared/      Cross-project utilities
projects/    One directory per portfolio project
tests/       Root tests for shared/
docs/        Architecture and project docs
```

See [docs/architecture/overview.md](docs/architecture/overview.md) for full details.
