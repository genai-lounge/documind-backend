# DocuMind Backend

The FastAPI backend for DocuMind, an educational RAG (Retrieval-Augmented Generation) app: upload documents, ask questions, and get Claude-generated answers with source citations you can check. The Next.js frontend lives in a separate repository (`documind-frontend`).

**Current status: Phase 2 (chunking and embeddings).** Uploaded PDF, DOCX, and TXT files are extracted, split into chunks, and each chunk is embedded by a configurable vendor: OpenAI (default), Voyage AI, or FastEmbed (local). A similarity endpoint compares any two texts. Nothing is stored yet, and there's no search or Claude functionality.

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
├── tests/                unit/ · contract/ · integration/ · api/ · live/
├── data/                 runtime data (uploads, Chroma); contents are git-ignored
└── docs/                 architecture, ADRs, product spec
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (it installs and manages Python 3.12)
- `make` (preinstalled on macOS and Linux), optional: each target just runs the `uv` command shown in the [Makefile](Makefile)

## Embedding provider setup (required to run the server)

Choose a vendor with `EMBEDDING_PROVIDER` in `backend/.env` (copy `.env.example` first). The server **refuses to start** if the chosen vendor isn't configured; the error says what's missing. Tests and CI never need a key.

| `EMBEDDING_PROVIDER` | Needs | Default model | Notes |
|---|---|---|---|
| `openai` (default) | `OPENAI_API_KEY` from [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | `text-embedding-3-small`, 1536 dimensions | Set `OPENAI_EMBEDDING_DIMENSIONS` (e.g. 512) to shorten vectors |
| `voyage` | `VOYAGE_API_KEY` from [dash.voyageai.com](https://dash.voyageai.com) | `voyage-4-lite`, 1024 dimensions | Without a payment method Voyage allows only ~3 requests/minute; add one (free tokens still apply). By default Voyage may train on API inputs: opt out in the dashboard (needs a payment method) and use test documents until then |
| `fastembed` | `uv sync --extra fastembed` (no key) | `BAAI/bge-small-en-v1.5`, 384 dimensions | Runs locally and offline. The first start downloads the model (~67 MB) to `data/models/` (git-ignored) |

`.env` is git-ignored; keys are never logged or sent to the browser.

**Vectors from different vendors or models can't be compared.** Changing provider or model is fine now (nothing is stored yet); from Phase 3 it will require re-embedding stored documents.

**Adding a vendor:** write an adapter in `app/infrastructure/embeddings/` (HTTP vendors can reuse `HttpEmbeddingClient` for timeouts, retries, and error handling), add a builder and one entry to `EMBEDDING_PROVIDERS` in `app/embedding_providers.py`, and add it to the contract tests. Nothing else changes.

## Run

```bash
uv sync      # create .venv and install locked dependencies
make dev     # start the API at http://localhost:8008
```

`make dev` runs `uv run uvicorn app.main:create_app --factory --reload --port 8008`. The app is built by the `create_app()` factory at startup, so importing the code has no side effects.

- Health check: `curl http://localhost:8008/api/v1/health` returns `{"status":"ok"}`.
- Interactive API docs: http://localhost:8008/docs
- Upload: `curl -F "file=@/path/to/report.pdf" http://localhost:8008/api/v1/documents`
- Compare two texts: `curl -H "Content-Type: application/json" -d '{"text_a":"car","text_b":"automobile"}' http://localhost:8008/api/v1/embeddings/similarity`

The frontend runs at http://localhost:3008.

## Checks

```bash
make check   # everything below; run before every commit
make lint    # ruff: style and common mistakes
make format  # ruff: fix formatting automatically
make typecheck  # mypy --strict
make arch    # import-linter: the architecture's dependency rules
make test    # pytest (all tests except live ones; no API key needed)
make test-live  # real vendors: each test skips itself unless that vendor's key (or the fastembed extra) is present
```

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the same checks on every pull request and on every push to `main`: `uv sync --locked`, then `make lint`, `make typecheck`, `make arch`, and `make test`. If `make check` passes locally, CI should pass too. See [ADR 0008](docs/adr/0008-continuous-integration.md).

To make a failing check **block merging**: on GitHub, open the repository's **Settings → Branches → Add branch protection rule** (or **Rules → Rulesets**) for `main`, enable **Require status checks to pass before merging**, and select **Lint, types, architecture, tests**.

All Python commands go through `uv run`. To add a dependency, use `uv add <pkg>` (or `uv add --dev <pkg>`), not pip.

## Configuration

Settings are read from environment variables and from `backend/.env` (loaded automatically). Every setting except the chosen vendor's API key has a default; empty values count as unset. Missing or invalid values stop the app at startup with a clear message.

| Variable | Default | Purpose |
|---|---|---|
| `MAX_UPLOAD_SIZE_MB` | `10` | Largest accepted upload (1–1024) |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:3008` | Browser origins allowed to call the API (comma-separated) |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` |
| `LOG_FORMAT` | `text` | `text` for terminals, `json` for log collectors |
| `EMBEDDING_PROVIDER` | `openai` | `openai`, `voyage`, or `fastembed` |
| `OPENAI_API_KEY` | none | Required when the provider is `openai` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `OPENAI_EMBEDDING_DIMENSIONS` | model default | Shorten text-embedding-3 vectors (at most the model's native size) |
| `VOYAGE_API_KEY` | none | Required when the provider is `voyage` |
| `VOYAGE_MODEL` | `voyage-4-lite` | Voyage embedding model |
| `VOYAGE_DIMENSIONS` | `1024` | 256, 512, 1024, or 2048 |
| `FASTEMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Any model FastEmbed supports |
| `FASTEMBED_CACHE_DIR` | `data/models` | Where local models are downloaded |
| `EMBEDDING_BATCH_SIZE` | `128` | Texts per request or model run (1–1000) |
| `EMBEDDING_TIMEOUT_SECONDS` | `30` | Read timeout per HTTP request |
| `EMBEDDING_MAX_RETRIES` | `3` | Retries on rate limits, server errors, and timeouts (0–10) |
| `CHUNK_SIZE` | `800` | Maximum characters per chunk (100–8000) |
| `CHUNK_OVERLAP` | `150` | Approximate characters shared by neighbouring chunks (less than `CHUNK_SIZE`) |

To override them, copy `.env.example` to `.env` and edit it. Never commit a real `.env` file.

Every response includes an `X-Request-ID` header, and every log line includes the same ID, so one request's logs can be found together.

## Troubleshooting

| Error code (HTTP 503) | Meaning | What to do |
|---|---|---|
| `embedding_rate_limited` | The vendor's per-minute limit was reached, even after retries | Wait a minute; for Voyage, add a payment method to raise the limits |
| `embedding_unavailable` | The vendor failed, timed out, rejected the key, or replied unexpectedly | Check the backend log for the request's ID: `embedding.unauthorized` means check the vendor's API key |

Each embedding call logs `embedding.completed` with the provider, model, batch count, `total_tokens` (what hosted vendors bill), and duration. Document text and the key are never logged.

## Known limitations (Phase 2)

- Uploaded files, chunks, and vectors are not stored; every upload is processed from scratch.
- If embedding fails, the whole upload fails (503). Per-document statuses arrive with storage.
- Uploads wait for Voyage, so large documents take longer; background processing is planned for Phase 7.
- Chunks never span pages, so a sentence that continues onto the next page is split. The last chunk on a page is often mostly overlap.
- Chunk sizes are counted in characters, not model tokens.
- No OCR: scanned PDFs return status `no_text`.
- DOCX files have no page numbers (Word only works them out when it displays a document), so their text is one section. Headers, footers, footnotes, and text boxes are not extracted.
- TXT files must be UTF-8.
- Password-protected PDFs are rejected.
- The size limit is checked after the upload arrives, so it protects parsing, not network bandwidth. A production deployment would also limit request size at the reverse proxy.
