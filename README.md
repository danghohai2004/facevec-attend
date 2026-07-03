r# Facial Recognition Attendance System Using Embeddings

In many enterprises, schools, and organizations, attendance tracking still relies on magnetic cards, fingerprint scanners, or manual entry. These approaches suffer from attendance fraud, slow operation, poor scalability, and weak integration with modern systems.

This project builds an automated **face-recognition attendance system** that operates in real time, is accurate, and scales for real-world deployment.

## Project Overview

The system is **embedding-based**: rather than image matching or training a per-person classifier, faces are turned into vectors and compared by similarity.

- Faces are processed with [InsightFace](https://github.com/deepinsight/insightface/tree/master/python-package) to extract a 512-dimensional embedding.
- Embeddings are stored in **[Qdrant](https://qdrant.tech/)** (a dedicated vector database) and searched by **cosine similarity**.
- Employee identity and attendance records live in **PostgreSQL**.

Advantages of the embedding approach:
- Fast recognition suitable for real-time use
- Scales to large numbers of users
- No retraining when new users are added
- Vector search handled by a purpose-built engine (HNSW index), not bolted onto the relational DB

> **Note on storage:** an earlier design stored vectors in PostgreSQL via pgvector. The current system uses **Qdrant** for vectors and **PostgreSQL** only for relational data. Face images are **not** persisted — they exist in memory only during registration, long enough to extract the embedding.

## Architecture at a Glance

```
Next.js dashboard ──(GET, direct)────────────►  FastAPI  ──►  PostgreSQL   (employees, attendance, shifts)
      │            ──(writes via BFF proxy)───►  :8000    ──►  Qdrant       (face_embeddings, cosine search)
      │                  + X-API-Key                 ▲
 (browser)                                           │  WebSocket frames → frame queue → recognition pipeline
                                          camera client (backend ready; kiosk UI is Phase 3)
```

- **Backend** — FastAPI **modular monolith** (`src/`), one process, domain-separated modules. Async SQLAlchemy (asyncpg) + `qdrant-client`. Managed with [uv](https://docs.astral.sh/uv/).
- **Recognition pipeline** — a WebSocket endpoint accepts JPEG frames into a bounded `asyncio` queue; a background task runs liveness → embedding extraction → Qdrant identify → attendance logging, then pushes the result back over the same socket.
- **Frontend** — Next.js (App Router) **admin dashboard**. Reads go straight to the backend; **writes go through a same-origin Backend-for-Frontend (BFF) proxy** that injects a server-only `X-API-Key`, so the secret never reaches the browser.
- **Vector DB** — Qdrant `face_embeddings` collection (512-dim, cosine, HNSW), payload `{emp_id, emp_code, name}` so a hit needs no join back to PostgreSQL.

## Recognition Model & Benchmarks

Multiple [InsightFace](https://github.com/deepinsight/insightface/tree/master/python-package) models were benchmarked CPU-only. The model is configured in `src/platform/config.py` (`MODEL`, `THRESHOLD`).

Test environment: AMD Ryzen 7 5800H, 16 GB RAM, CPU-only inference, every 5th frame, averaged over 300 runs.

| Model | End-to-End Inference Time |
|-------|---------------------------|
| `buffalo_sc` | **~19.9 ms** |
| `buffalo_s` | ~100 ms |
| `buffalo_m` | ~175 ms |

**Conclusion:** `buffalo_sc` gives the lowest latency and is the default for real-time, CPU-based use.

## Repository Layout

```
facevec-attend/
├─ main.py                  # FastAPI entry point (uvicorn)
├─ compose.yaml             # PostgreSQL + Qdrant services
├─ initdb/init.sql          # PostgreSQL schema + seed (auto-applied on first boot)
├─ pyproject.toml / uv.lock # Python deps (uv)
├─ Makefile                 # install / db-up / db-down / run / clean
├─ scripts/
│  └─ reconcile_vectors.py  # report PG↔Qdrant drift (read-only)
├─ src/
│  ├─ app.py                # app factory, lifespan, router wiring, pipeline task
│  ├─ platform/             # shared infra (no business logic)
│  │  ├─ config.py          #   MODEL + THRESHOLD
│  │  ├─ auth.py            #   require_api_key (fail-closed X-API-Key)
│  │  ├─ queue.py           #   FrameQueue (bounded, drop-oldest)
│  │  ├─ db/                #   session (Postgres async) + qdrant client
│  │  ├─ ml/face_app.py     #   InsightFace singleton
│  │  └─ realtime/manager.py#   WebSocket connection registry
│  └─ modules/
│     ├─ employees/         # CRUD + face registration
│     ├─ attendance/        # check-in/out, shift settings
│     ├─ recognition/       # ws_ingress, pipeline, extractor, identifier
│     ├─ antispoofing/      # LivenessChecker interface + pass-through default
│     └─ tts/               # offline Piper voice for the kiosk's audio greeting
├─ models/piper/            # self-hosted Piper voice model (vi_VN, ~63MB)
├─ frontend/                # Next.js admin dashboard + BFF write proxy
├─ tests/                   # pytest suite (mirrors src/ layout)
└─ docs/                    # detailed documentation (start here)
```

## Documentation

| Doc | What it covers |
|-----|----------------|
| [docs/setup/getting-started.md](docs/setup/getting-started.md) | Install, configure `.env`, run backend + frontend + databases |
| [docs/architecture/overview.md](docs/architecture/overview.md) | Components, recognition pipeline, security model, scalability |
| [docs/api/api_spec.md](docs/api/api_spec.md) | REST + WebSocket endpoints, auth, request/response shapes |
| [docs/data/schema.md](docs/data/schema.md) | PostgreSQL tables + Qdrant collection |
| [docs/security/follow-ups.md](docs/security/follow-ups.md) | Known security debt and deferred hardening |

## Quick Start

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker + Docker Compose, Node.js 20+.

```bash
# 1. configure environment (set DB_PASS, QDRANT_API_KEY, API_KEY)
cp .env.example .env

# 2. start PostgreSQL + Qdrant
make db-up

# 3. run the backend (http://localhost:8000/docs)
uv sync && make run

# 4. run the dashboard (http://localhost:3000 → /dashboard)
cd frontend && npm install && npm run dev
```

See [docs/setup/getting-started.md](docs/setup/getting-started.md) for the full walkthrough, including the frontend BFF environment variables (`API_KEY`, `BACKEND_INTERNAL_URL`) required for write actions.

## Project Status

- ✅ Backend: employees CRUD + face registration, attendance (shift-window enforced), shift settings, recognition WebSocket pipeline, Qdrant vector search.
- ✅ Security: write endpoints require `X-API-Key`; frontend BFF proxy injects it server-side; production refuses to boot with the placeholder liveness checker.
- ✅ Frontend: admin dashboard — employees, shifts, and manual attendance are wired to the real API; the dashboard landing page currently shows placeholder metrics.
- ⏳ Deferred (Phase 3): a live kiosk client (camera + WebSocket) in the browser, real liveness detection, user login/RBAC + audit, and TLS. See [docs/security/follow-ups.md](docs/security/follow-ups.md).

## License

See [LICENSE](LICENSE).
