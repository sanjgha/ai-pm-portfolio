# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Monorepo Structure

```
shared/          Cross-project utilities (logging, config helpers)
projects/        One directory per portfolio project (each is an independent Python package)
tests/           Root-level tests that cover shared/
docs/            Architecture specs and project briefs
```

Each project under `projects/<slug>/` has its own `pyproject.toml`, `Makefile`, `src/`, and `tests/`. The root `pyproject.toml` governs tooling for `shared/` and `tests/` only.

## Common Commands

**Root (shared + tests only):**
```bash
make dev-install      # install root dev toolchain
make ci               # lint + test shared/
make lint             # ruff check + format check
make format           # auto-format
make typecheck        # mypy shared/
make test             # pytest tests/
make test-cov         # pytest with HTML coverage report
```

**Monorepo-wide:**
```bash
make ci-all           # CI across every project
make test-all         # tests across every project
make lint-all         # lint across every project
```

**Single test file:**
```bash
pytest tests/unit/test_foo.py -v
```

**Scaffold a new project:**
```bash
make new-project name=<slug> desc="One-line description"
cd projects/<slug> && pip install -e ".[dev]" && pre-commit install
```

## Guardrails

| Guard | Tool | Scope |
|-------|------|-------|
| Lint + format | ruff (line-length 100) | root + per-project |
| Type checking | mypy (strict=false, py3.10) | root + per-project |
| Tests + coverage | pytest-cov | root + per-project |
| Secret scanning | detect-secrets | pre-commit |
| Branch protection | pre-push hook | blocks direct push to main/master |

mypy runs as a **manual** pre-commit stage: `pre-commit run mypy --all-files`. It does not run on every commit automatically.

## Branch Workflow

Direct pushes to `main`/`master` are blocked by `.githooks/pre-push`. Always work on a feature branch and PR into `main`.

## Per-Project Pattern

Each project under `projects/` is a standalone Python package with `src/` layout. When working inside a project, run its own `make ci` / `pytest` rather than the root ones — they have separate venvs and dependencies.

## Linear-First Workflow

All work tracks through Linear. Hierarchy:

```
Project (new-project.md)
  ├─ Initiative/Epic
  │   └─ Issue (new-issue.md, enhancement.md, bug-report.md)
  │       └─ Subtask (subtask.md)
```

**Every issue must include:**
- `Project Context` — which Linear project and parent issue
- `Classification` — priority, labels, assignee
- `Definition of Done` — explicit completion checklist

## Workflow

- **New projects:** Start with `new-project.md` to initialize a project in Linear with milestones, team, and success criteria.
- **Features/Bugs:** Create issues using `new-issue.md`, `bug-report.md`, or `enhancement.md`.
- **Subtasks:** For smaller work items under a parent issue, use `subtask.md`.
- **Follow TDD:** Write a failing test first, then implement to make it pass.
- **Format before committing:** Run `ruff format src/ tests/` to auto-format code. Pre-commit hooks will also auto-fix lint issues.
- **Commit messages** must follow Conventional Commits (`feat:`, `fix:`, `test:`, `refactor:`, `chore:`) — enforced by pre-commit hooks.

### Workflow Isolation

- **One worktree per task.** Every feature/fix runs in its own worktree branched from `master` (via the `using-git-worktrees` skill / `EnterWorktree`). The main checkout at the repo root is for browsing only — never accumulate feature work there. This keeps `git status` in the worktree equal to exactly the change set for that PR.
- **Keep volatile files out of git.** Scraped data and generated files should be gitignored. Only curated references and source code are tracked.

### Task Tracking with TodoWrite

**Use TodoWrite/TaskCreate when:**
- Task involves **4+ steps** (more than 3 discrete actions)
- Task will use **6+ tools** (more than 5 tool calls)

**Required final step:** Always add a verification step at the end that checks all previous items for completion before claiming work is done.

Example:
```
1. Implement X
2. Add tests for X
3. Update documentation
4. ✅ Verify: Run tests, check coverage, confirm docs updated
```
