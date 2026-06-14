.PHONY: install dev-install test test-cov lint format typecheck ci clean ingest ingest-all check-updates reingest serve-api serve-ui eval

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

# Ingest targets
ingest:
	python -m exchange_connectivity_hub.ingest ingest --doc $(doc)

ingest-all:
	python -m exchange_connectivity_hub.ingest ingest-all

check-updates:
	python -m exchange_connectivity_hub.ingest.update_checker

reingest:
	python -m exchange_connectivity_hub.ingest reingest --doc $(doc)

# API and UI
serve-api:
	uvicorn exchange_connectivity_hub.api.main:app --reload --host 0.0.0.0 --port 8000

serve-ui:
	streamlit run exchange_connectivity_hub/ui/app.py

# Eval
eval:
	python evals/run_eval.py
