# Shortcuts for common backend tasks. Run `make <target>` from backend/.
# Each target just runs the uv commands shown, so you can also run them directly.

.PHONY: dev test test-live lint format typecheck arch check

dev:        ## Start the API with auto-reload at http://localhost:8008
	uv run uvicorn app.main:create_app --factory --reload --port 8008

test:       ## Run all tests except live ones (no API keys needed)
	uv run pytest

test-live:  ## Run tests against real external services (needs VOYAGE_API_KEY)
	uv run pytest -m live

lint:       ## Check code style and common mistakes
	uv run ruff check .
	uv run ruff format --check .

format:     ## Fix formatting and auto-fixable lint issues
	uv run ruff check --fix .
	uv run ruff format .

typecheck:  ## Check type annotations (strict)
	uv run mypy

arch:       ## Check the architecture's dependency rules
	uv run lint-imports

check: lint typecheck arch test  ## Everything CI would run; use before every commit
