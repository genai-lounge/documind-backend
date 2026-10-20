# DocuMind Backend

The FastAPI backend for DocuMind, an educational RAG (Retrieval-Augmented Generation) app: upload documents, ask questions, and get Claude-generated answers with source citations you can check. The Next.js frontend lives in a separate repository (`documind-frontend`).

**Current status: Phase 0 (project setup only).** There's no document, embedding, vector-store, or Claude functionality yet. See [docs/architecture.md](docs/architecture.md) for the plan and [docs/documind_product_requirements_and_phases.md](docs/documind_product_requirements_and_phases.md) for the full spec.

## Structure

```text
backend/
├── app/       FastAPI application (app/main.py is the entry point)
├── tests/     pytest tests
├── data/      Runtime data (uploads, Chroma). Contents are git-ignored.
└── docs/      Architecture notes and product spec
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (it installs and manages Python 3.12)

## Run

```bash
uv sync                                            # create .venv and install locked dependencies
uv run uvicorn app.main:app --reload --port 8008   # start the API at http://localhost:8008
uv run pytest                                      # run tests
uv run ruff check .                                # lint
```

Health check: `curl http://localhost:8008/api/v1/health` returns `{"status":"ok"}`.
Interactive API docs: http://localhost:8008/docs

All Python commands go through `uv run`. To add a dependency, use `uv add <pkg>` (or `uv add --dev <pkg>`), not pip.

The frontend runs at http://localhost:3008.

## Configuration

`.env.example` lists the environment variables. Phase 0 doesn't need any. Copy it to `.env` when a later phase needs one. Never commit a real `.env` file.
