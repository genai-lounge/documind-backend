# DocuMind Backend

The FastAPI backend for DocuMind, an educational RAG (Retrieval-Augmented Generation) app: upload documents, ask questions, and get Claude-generated answers with source citations you can check. The Next.js frontend lives in a separate repository (`documind-frontend`).

**Current status: Phase 4 (answers with Claude).** Uploaded PDF, DOCX, and TXT files are extracted, chunked, embedded (OpenAI by default, Voyage AI, or FastEmbed), and stored (files on disk or in Supabase Storage, records in SQLite, vectors in Chroma local or Chroma Cloud). You can **ask questions**: the closest passages are retrieved and Claude answers from them only, with citations that are checked against the passages actually retrieved; when the documents don't answer the question, it says so.

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
├── data/                 runtime data, git-ignored: uploads/ (original files),
│                         documind.db (SQLite records), chroma/ (vectors), models/ (FastEmbed)
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

**Vectors from different vendors or models can't be compared.** Each model gets its own vector collection, so switching provider or model is safe, but documents indexed with the old model aren't searchable until they're uploaded again (they show `searchable: false`).

**Adding a vendor:** write an adapter in `app/infrastructure/embeddings/` (HTTP vendors can reuse `HttpEmbeddingClient` for timeouts, retries, and error handling), add a builder and one entry to `EMBEDDING_PROVIDERS` in `app/embedding_providers.py`, and add it to the contract tests. Nothing else changes.

## Claude setup (required to run the server)

Answers are written by Claude through Anthropic's official Python SDK. The server **refuses to start without `ANTHROPIC_API_KEY`**; tests and CI never need it.

1. Create a key at [console.anthropic.com](https://console.anthropic.com) → API Keys.
2. Add `ANTHROPIC_API_KEY=sk-ant-...` to `backend/.env` (never to `.env.example`, which is committed).

| Setting | Default | Notes |
|---|---|---|
| `ANTHROPIC_MODEL` | `claude-haiku-5-5` | Fast and inexpensive ($0.10 / $0.50 per million input / output tokens). `claude-sonnet-5-5` ($2 / $10) or `claude-opus-5-5` ($4 / $20) for harder questions |
| `ANTHROPIC_EFFORT` | `medium` | `low`, `medium`, `high`, `xhigh`, `max`: how much the model reasons before answering; the main cost lever |
| `ANSWER_TOP_K` | `5` | Passages given to Claude per question |

**How an answer is made:** the question retrieves the `ANSWER_TOP_K` closest chunks; each is sent to Claude as a `search_result` block (source = chunk ID, title = file and page) with citations enabled, under a system prompt that requires answering from those passages only, saying when they don't contain the answer, and treating document text as data, never as instructions. Every citation Claude returns is checked: it must point at a passage that was actually sent and quote text that is really in it; anything else is dropped and logged as `dropped_citations`. An answer with no valid citation is marked `insufficient_evidence`. With no indexed documents, Claude isn't called at all. The question and answer text are never logged.

## File storage setup: local or Supabase

`FILE_STORAGE` chooses where original uploaded files are kept. On upload, the file is saved **before** anything is embedded or written to Chroma; if saving fails, the document is marked `failed` and nothing else is written.

| `FILE_STORAGE` | Needs | Notes |
|---|---|---|
| `local` (default) | nothing | `DATA_DIR/uploads/` |
| `supabase` | `SUPABASE_URL` (Project Settings → API) and `SUPABASE_SECRET_KEY` (Project Settings → API Keys → secret key, `sb_secret_...`; the legacy `service_role` key also works) | Files go to a **private** bucket, `SUPABASE_BUCKET` (default `documents`), which the backend **creates at startup if it's missing** |

The secret key bypasses Supabase's row-level security: keep it **only in `backend/.env`**, never in the frontend. On Supabase's free plan a project pauses after about a week of inactivity; the backend then refuses to start with a clear message until you resume it in the dashboard. Switching `FILE_STORAGE` doesn't move existing files. Tests and CI always use local storage; `make test-live` includes one tiny upload/overwrite/delete in your bucket when `FILE_STORAGE=supabase`.

## Vector store setup: local or Chroma Cloud

`CHROMA_MODE` chooses where chunk vectors are stored. Upload, search, and delete work the same either way.

| `CHROMA_MODE` | Needs | Notes |
|---|---|---|
| `local` (default) | nothing | Embedded Chroma in `DATA_DIR/chroma/`; no network or key; one server process |
| `cloud` | `CHROMA_API_KEY` from your [Chroma Cloud](https://www.trychroma.com) dashboard; `CHROMA_TENANT` and `CHROMA_DATABASE` only if the key covers more than one database | Managed, shared by several server instances. The connection is checked at startup, so bad credentials stop the server with a clear message |

With Chroma Cloud, **chunk text and vectors are stored on Chroma's servers** and every write, search, and stored gigabyte is billed against your Chroma credits. Use test documents until you're comfortable with that. Tests and CI always use local Chroma, so they never touch your account; `make test-live` includes one small Chroma Cloud round trip (a throwaway collection, deleted afterwards) when `CHROMA_MODE=cloud`.

Switching mode doesn't copy data: vectors stored in one place aren't visible in the other, so upload documents again after switching.

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

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/documents` | Upload: extract, chunk, embed, store, and index (`index_status` in the response) |
| GET | `/api/v1/documents` | Library: every document with status, chunk count, and embedding model |
| GET | `/api/v1/documents/{id}` | One document with its stored chunks |
| DELETE | `/api/v1/documents/{id}` | Remove the document's vectors, file, and record (204; 404 if unknown) |
| POST | `/api/v1/retrieval/search` | `{"query", "top_k"}`: the closest chunks, with rank, score, file, and page |
| POST | `/api/v1/embeddings/similarity` | Compare two texts |
| POST | `/api/v1/chat` | `{"question"}`: Claude's answer with `[n]` citation markers, the checked `citations`, every retrieved passage (`sources`), and `insufficient_evidence` |

Ask from the command line: `curl -H "Content-Type: application/json" -d '{"question":"How many vacation days do employees get?"}' http://localhost:8008/api/v1/chat`

Search from the command line: `curl -H "Content-Type: application/json" -d '{"query":"How much did revenue grow?","top_k":3}' http://localhost:8008/api/v1/retrieval/search`

The score is cosine similarity: it ranks results; it is **not** the probability that a passage answers the question.

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
| `DATA_DIR` | `backend/data` | Where files, the SQLite database, and local Chroma are stored |
| `FILE_STORAGE` | `local` | `local` (disk) or `supabase` (Supabase Storage) |
| `SUPABASE_URL` | none | Required when `FILE_STORAGE=supabase`, e.g. `https://abcd.supabase.co` |
| `SUPABASE_SECRET_KEY` | none | Required when `FILE_STORAGE=supabase`; backend only |
| `SUPABASE_BUCKET` | `documents` | Private bucket for uploads; created if missing |
| `CHROMA_MODE` | `local` | `local` (embedded, on disk) or `cloud` (Chroma Cloud) |
| `CHROMA_API_KEY` | none | Required when `CHROMA_MODE=cloud` |
| `CHROMA_TENANT` / `CHROMA_DATABASE` | from the key | Only needed when the key covers several databases |
| `SEARCH_DEFAULT_TOP_K` | `5` | Results returned when `top_k` isn't given |
| `SEARCH_MAX_TOP_K` | `20` | Largest `top_k` a search may ask for |
| `CHUNK_SIZE` | `800` | Maximum characters per chunk (100–8000) |
| `CHUNK_OVERLAP` | `150` | Approximate characters shared by neighbouring chunks (less than `CHUNK_SIZE`) |

To override them, copy `.env.example` to `.env` and edit it. Never commit a real `.env` file.

Every response includes an `X-Request-ID` header, and every log line includes the same ID, so one request's logs can be found together.

## Troubleshooting

| Error code (HTTP 503) | Meaning | What to do |
|---|---|---|
| `embedding_rate_limited` | The vendor's per-minute limit was reached, even after retries | Wait a minute; for Voyage, add a payment method to raise the limits |
| `embedding_unavailable` | The vendor failed, timed out, rejected the key, or replied unexpectedly | Check the backend log for the request's ID: `embedding.unauthorized` means check the vendor's API key |
| `llm_unavailable` | Claude failed, timed out, rejected the key or request, or the answer was cut off | Check the log for `llm.unauthorized` (check `ANTHROPIC_API_KEY`), `llm.rejected` (e.g. a wrong `ANTHROPIC_MODEL`), or `llm.unreachable` |
| `llm_rate_limited` | Anthropic rate limit reached after the SDK's retries | Wait (see the `Retry-After` header) |
| `answer_refused` (HTTP 422) | Claude's safety checks declined the question | Rephrase it. Claude Haiku has no automatic fallback model |
| `storage_unavailable` | The file store, SQLite, or Chroma failed | Check the log for `file_storage.*`, `document_store.failed`, or `vector_store.failed`, and that `DATA_DIR` is writable |

A document whose indexing failed stays in the library with status `failed` and an `error_code`, so failures are never silent. Its original file is kept. Delete it and upload again.

Each embedding call logs `embedding.completed` with the provider, model, batch count, `total_tokens` (what hosted vendors bill), and duration. Document text and the key are never logged.

## Known limitations (Phase 4)

- **Changing the embedding provider or model** puts new documents in a separate vector collection; documents indexed with the old model show `searchable: false` and must be deleted and uploaded again (automatic re-indexing is planned for Phase 7).
- **One server process with local Chroma** (embedded in the app). With `CHROMA_MODE=cloud`, several instances can share the vectors, and with `FILE_STORAGE=supabase` so can the files, but document records (SQLite) are still per machine; a Postgres adapter (for example Supabase's database) would complete that.
- Switching `CHROMA_MODE` doesn't migrate vectors; upload documents again.
- **No transactions across stores**: if the server crashes mid-upload, a record can stay `processing`. Repair and retries come in Phase 7.
- Answers are single questions: no conversation history or follow-ups yet, and the answer arrives all at once rather than streaming (Phase 5).
- Each question costs one embedding call plus one Claude call; with Claude Haiku 5.5 a typical question costs well under a cent.
- Uploads wait for embedding and indexing to finish (background processing in Phase 7).
- Chunks never span pages; the last chunk on a page is often mostly overlap; sizes are counted in characters, not tokens.
- No OCR; TXT must be UTF-8; password-protected PDFs are rejected.
- The upload size limit protects parsing, not bandwidth; production would also limit request size at the reverse proxy.
