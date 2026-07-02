# Frontend — Face Recognition Attendance Dashboard

Next.js (App Router) **admin dashboard** for the attendance system. It runs as a
Node server (`next start`, not a static export) so it can host server-side Route
Handlers — specifically the write proxy described below.

> Not a kiosk. The live camera / WebSocket recognition client is Phase 3 (see
> [`../docs/security/follow-ups.md`](../docs/security/follow-ups.md)). Today this
> app manages employees, shift settings, and manual attendance.

## Pages

| Route | Purpose | Data |
|-------|---------|------|
| `/` → `/dashboard` | overview / KPIs | placeholder (mock) |
| `/employees` | list, register (with face photo), delete | real API |
| `/attendance` | manual check-in/out by employee id | real API |
| `/shifts` | view / update shift windows | real API |

## How it talks to the backend

- **Reads (GET)** go straight to the backend at `NEXT_PUBLIC_API_BASE_URL`
  (default `http://localhost:8000`).
- **Writes (POST/PUT/DELETE)** go through a same-origin **BFF proxy** at
  `src/app/api/write/[...path]/route.ts`. The proxy injects a server-only
  `X-API-Key` and forwards to `BACKEND_INTERNAL_URL`, so the API key never
  reaches the browser. It also blocks cross-origin writes (CSRF guard).

## Environment (`.env.local`)

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000   # public: read base URL

# server-only — never prefix with NEXT_PUBLIC_
API_KEY=<must equal the backend's API_KEY>
BACKEND_INTERNAL_URL=http://localhost:8000
```

If `API_KEY` here does not match the backend, all write actions return 401.

## Develop

```bash
npm install
npm run dev        # http://localhost:3000
```

```bash
npm run lint
npx tsc --noEmit
npm run build
```

See [`../docs/setup/getting-started.md`](../docs/setup/getting-started.md) for the
full stack (backend + databases).
