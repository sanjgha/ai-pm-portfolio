.PHONY: install dev-install test test-cov lint format typecheck ci clean serve-api serve-ui eval

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

serve-api:
	uvicorn pre_trade_risk_rules_assistant.api.main:app --reload --host 0.0.0.0 --port 8000

serve-ui:
	streamlit run src/pre_trade_risk_rules_assistant/ui/app.py

eval:
	python evals/run_eval.py
