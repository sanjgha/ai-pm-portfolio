# Portfolio Architecture

## Monorepo Layout

```
ai-pm-portfolio/
├── shared/              # Cross-project utilities (logging, config helpers, etc.)
├── projects/            # One directory per portfolio project
│   ├── <project-slug>/
│   │   ├── pyproject.toml   # Own deps + build config
│   │   ├── Makefile         # make ci / test / lint
│   │   ├── src/             # Application code
│   │   ├── tests/           # Unit + integration
│   │   └── README.md        # Project brief
│   └── ...
├── tests/               # Root-level tests for shared/
├── docs/                # Architecture + project docs
├── Makefile             # Root orchestrator
└── pyproject.toml       # Root tooling (ruff, mypy, pytest settings)
```

## Adding a New Project

```bash
make new-project name=my-project desc="What it does"
```

This scaffolds `projects/my-project/` with a full Python toolchain
(ruff, mypy, pytest, pre-commit, detect-secrets).

## Guardrails in Place

| Guard | Tool | Where |
|-------|------|--------|
| Linting + formatting | ruff | per-project + root |
| Type-checking | mypy | per-project + root |
| Tests + coverage | pytest-cov | per-project + root |
| Secret scanning | detect-secrets | pre-commit hook |
| Branch protection | pre-push hook | `.githooks/pre-push` |
| Pre-commit suite | pre-commit | root `.pre-commit-config.yaml` |
