# DocuMind Backend

The FastAPI backend for DocuMind, an educational RAG (Retrieval-Augmented Generation) app: upload documents, ask questions, and get Claude-generated answers with source citations you can check. The Next.js frontend lives in a separate repository (`documind-frontend`).

**Current status: Phase 1 (document upload and text extraction), on a clean architecture.** You can upload PDF, DOCX, and TXT files and inspect the extracted text and metadata. Nothing is stored yet, and there's no chunking, embedding, vector-store, or Claude functionality.

- [docs/architecture.md](docs/architecture.md): how the system is designed, with layers, ports, and request flow
- [docs/adr/](docs/adr/README.md): why it is designed that way (Architecture Decision Records)
- [docs/documind_product_requirements_and_phases.md](docs/documind_product_requirements_and_phases.md): the full product spec

## Structure

```text
backend/
├── app/
│   ├── domain/           core concepts and errors (pure Python, no frameworks)
│   ├── application/      use cases + ports (interfaces the app needs)
│   ├── infrastructure/   adapters: vendor libraries live only here
│   ├── core/             settings and logging setup
│   ├── api/              FastAPI routes, HTTP schemas, error mapping, middleware
│   ├── container.py      composition root: wires adapters into use cases
│   └── main.py           create_app() factory and the uvicorn entry point
├── tests/                unit/ · contract/ · integration/ · api/
├── data/                 runtime data (uploads, Chroma); contents are git-ignored
└── docs/                 architecture, ADRs, product spec
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (it installs and manages Python 3.12)
- `make` (preinstalled on macOS and Linux), optional: each target just runs the `uv` command shown in the [Makefile](Makefile)

## Run

```bash
uv sync      # create .venv and install locked dependencies
make dev     # start the API at http://localhost:8008 (uv run uvicorn app.main:app --reload --port 8008)
```

- Health check: `curl http://localhost:8008/api/v1/health` returns `{"status":"ok"}`.
- Interactive API docs: http://localhost:8008/docs
- Upload from the command line: `curl -F "file=@/path/to/report.pdf" http://localhost:8008/api/v1/documents`

The frontend runs at http://localhost:3008.

## Checks

```bash
make check   # everything below; run before every commit
make lint    # ruff: style and common mistakes
make format  # ruff: fix formatting automatically
make typecheck  # mypy --strict
make arch    # import-linter: the architecture's dependency rules
make test    # pytest
```

All Python commands go through `uv run`. To add a dependency, use `uv add <pkg>` (or `uv add --dev <pkg>`), not pip.

## Configuration

Settings are read from environment variables and from `backend/.env` (loaded automatically). Every setting has a default, so `.env` is optional. Invalid values stop the app at startup with a clear message.

| Variable | Default | Purpose |
|---|---|---|
| `MAX_UPLOAD_SIZE_MB` | `10` | Largest accepted upload (1–1024) |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:3008` | Browser origins allowed to call the API (comma-separated) |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` |
| `LOG_FORMAT` | `text` | `text` for terminals, `json` for log collectors |

To override them, copy `.env.example` to `.env` and edit it. Never commit a real `.env` file.

Every response includes an `X-Request-ID` header, and every log line includes the same ID, so one request's logs can be found together.

## Known limitations (Phase 1)

- Uploaded files and results are not stored.
- No OCR: scanned PDFs return status `no_text`.
- DOCX files have no page numbers (Word only works them out when it displays a document), so their text is one section. Headers, footers, footnotes, and text boxes are not extracted.
- TXT files must be UTF-8.
- Password-protected PDFs are rejected.
- The size limit is checked after the upload arrives, so it protects parsing, not network bandwidth. A production deployment would also limit request size at the reverse proxy.
