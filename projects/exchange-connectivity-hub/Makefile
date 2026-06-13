.PHONY: install dev-install test test-cov lint format typecheck ci clean

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=src --cov-report=html

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	ruff format src/ tests/

typecheck:
	mypy src/ --ignore-missing-imports

ci: lint test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache build/ dist/ *.egg-info htmlcov/ .coverage
