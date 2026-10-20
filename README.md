# DocuMind Backend

The FastAPI backend for DocuMind, an educational RAG (Retrieval-Augmented Generation) app: upload documents, ask questions, and get Claude-generated answers with source citations you can check. The Next.js frontend lives in a separate repository (`documind-frontend`).

**Current status: Phase 1 (document upload and text extraction).** You can upload PDF, DOCX, and TXT files and inspect the extracted text and metadata. Nothing is stored yet, and there's no chunking, embedding, vector-store, or Claude functionality. See [docs/architecture.md](docs/architecture.md) for the plan and [docs/documind_product_requirements_and_phases.md](docs/documind_product_requirements_and_phases.md) for the full spec.

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
uv run ruff format .                               # format code
```

Health check: `curl http://localhost:8008/api/v1/health` returns `{"status":"ok"}`.
Interactive API docs: http://localhost:8008/docs (you can try the upload endpoint there)

Upload a file from the command line:

```bash
curl -F "file=@/path/to/report.pdf" http://localhost:8008/api/v1/documents
```

All Python commands go through `uv run`. To add a dependency, use `uv add <pkg>` (or `uv add --dev <pkg>`), not pip.

The frontend runs at http://localhost:3008.

## Configuration

`.env.example` lists the environment variables. Every variable has a default, so no `.env` file is required yet.

| Variable | Default | Purpose |
|---|---|---|
| `MAX_UPLOAD_SIZE_MB` | `10` | Largest accepted upload |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:3008` | Browser origins allowed to call the API (comma-separated) |

To override them, copy `.env.example` to `.env` and start with `uv run --env-file .env uvicorn app.main:app --reload --port 8008`.

Never commit a real `.env` file.

## Known limitations (Phase 1)

- Uploaded files and results are not stored.
- No OCR: scanned PDFs return status `no_text`.
- DOCX files have no page numbers (Word only works them out when it displays a document), so their text is one section. Headers, footers, footnotes, and text boxes are not extracted.
- TXT files must be UTF-8.
- Password-protected PDFs are rejected.
- The size limit is checked after the upload arrives, so it protects parsing, not network bandwidth. A production deployment would also limit request size at the reverse proxy.
