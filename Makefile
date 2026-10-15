.PHONY: setup test test-api run lint clean help

help:
	@echo "Available commands:"
	@echo "  make setup     - uv sync --extra dev (resolve + create .venv)"
	@echo "  make test      - run all unit tests (MOCK_LLM=1)"
	@echo "  make test-api  - run API suite"
	@echo "  make run       - start FastAPI uvicorn server"
	@echo "  make lint      - check code quality with ruff"
	@echo "  make clean     - remove caches, .venv, and build artifacts"

setup:
	uv sync --extra dev

test:
	MOCK_LLM=1 uv run pytest -v

test-api:
	MOCK_LLM=1 uv run pytest -v tests/test_api.py

run:
	MOCK_LLM=1 uv run uvicorn src.parte4_api.main:app --reload --port 8000

lint:
	uv run ruff check src/ tests/ || true

clean:
	rm -rf .venv **/__pycache__ .pytest_cache .ruff_cache build dist *.egg-info
