.PHONY: install dev-install test test-cov lint format typecheck ci clean gen-data serve eval

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=src --cov-report=html

lint:
	ruff check src/ tests/ evals/
	ruff format --check src/ tests/ evals/
	mypy src/ --ignore-missing-imports

format:
	ruff format src/ tests/ evals/

typecheck:
	mypy src/ --ignore-missing-imports

ci: lint test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache build/ dist/ *.egg-info htmlcov/ .coverage

gen-data:
	python -m execution_quality_copilot.datagen

serve:
	python -m execution_quality_copilot.server.main

eval:
	python evals/run_eval.py
