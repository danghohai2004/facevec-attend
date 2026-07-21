# Source Code Review Report

## Executive Summary

* Overall health: **Not production-ready.** The modular backend structure is reasonable, but core security, biometric integrity, frontend/backend integration, and operational safety have major gaps.
* Biggest risks: unauthenticated destructive APIs, fake frontend recognition, disabled liveness protection, destructive database shutdown command, unbounded recognition workload, and PostgreSQL/Qdrant consistency failures.
* Recommended priority: secure and repair the attendance path first; then bound processing and uploads; then add migrations, integration tests, and deployment gates.
* Verification:
  * Backend: **10 tests passed**.
  * Frontend: **ESLint and TypeScript checks passed**.
  * Build: not run because Next build writes generated files, conflicting with the no-modification requirement.
  * npm audit: **2 high and 1 moderate advisories**.
  * Worktree remained unchanged except for the pre-existing `README.md` modification.

## Critical Issues

### 1. `make db-down` deletes persistent attendance data

* File: `Makefile:25`
* Problem: `db-down` runs `docker compose down -v`, although its help text describes only stopping services. `-v` deletes PostgreSQL and Qdrant volumes.
* Impact: An ordinary shutdown command can irreversibly erase employees, attendance history, shift configuration, and face embeddings.
* Suggested fix: Make `db-down` run `docker compose down`. Add a separately named, confirmation-protected command such as `db-reset` for volume deletion.
* Confidence: High.

### 2. Administrative and biometric APIs have no authentication or authorization

* File: `src/modules/employees/api.py:26`, `src/modules/attendance/api.py:17`, `src/modules/recognition/ws_ingress.py:12`, `main.py:7`
* Problem: Anyone reaching port 8000 can register or replace biometrics, delete employees, inspect attendance, alter shifts, forge attendance, or open recognition WebSockets. The server binds to all interfaces.
* Impact: Unauthorized biometric enrollment, attendance fraud, personal-data disclosure, and destructive administrative actions.
* Suggested fix: Separate kiosk and administrator trust boundaries. Require authenticated roles for employee, history, and shift operations; authenticate kiosk devices or use short-lived signed credentials.
* Confidence: High for the code; external exploitability depends on firewall/network isolation and needs verification.

### 3. Frontend recognition is simulated and cannot perform the documented workflow

* File: `frontend/src/components/attendance/attendance-client.tsx:43`, `frontend/src/lib/api.ts:216`
* Problem: The UI selects a random demo employee every two seconds and displays that as a detection. It never connects to `/ws/recognition/{client_id}` or sends captured frames. Its REST calls put `emp_id` in a JSON body, while FastAPI requires an integer query parameter; generated OpenAPI confirms there is no request body.
* Impact: Users see false identities, and check-in/check-out requests return HTTP 422. The primary attendance flow is nonfunctional and misleading.
* Suggested fix: Remove simulated detection from production UI. Implement the documented binary WebSocket flow and consume server recognition results. Align the manual attendance client with the generated API contract.
* Confidence: High.

### 4. Anti-spoofing always approves every frame

* File: `src/modules/antispoofing/service.py:10`
* Problem: `PassThroughChecker.check()` always returns `True`.
* Impact: A photograph, replayed video, or screen image can potentially generate attendance. The core biometric trust claim is not enforced.
* Suggested fix: Keep the interface, but block production deployment until a tested liveness implementation and spoofing acceptance thresholds are configured.
* Confidence: High.

## High Priority Issues

### 1. Upload processing is unbounded and permits biometric poisoning

* File: `src/modules/employees/api.py:36`, `src/modules/recognition/extractor.py:12`
* Problem: Uploads are read completely into memory without size, MIME, image-dimension, or magic-byte limits. Every face found in the image is enrolled under the same employee.
* Impact: Memory/decompression denial of service; bystanders or deliberately included faces become valid identities for another employee.
* Suggested fix: Enforce request and decoded-image limits, validate actual image format, require exactly one acceptable face per enrollment frame, and reject ambiguous images.
* Confidence: High.

### 2. Frontend captures up to 20 files, but backend accepts one scalar file

* File: `frontend/src/components/employees/employee-registration.tsx:134`, `frontend/src/lib/api.ts:152`, `src/modules/employees/api.py:30`
* Problem: The frontend appends multiple `file` fields, while FastAPI’s generated schema declares one binary `file`.
* Impact: Most captured enrollment frames are ignored or ambiguously parsed, reducing recognition quality while the UI reports successful multi-frame enrollment.
* Suggested fix: Define an explicit `list[UploadFile]` contract with count/size limits, or change the UI to send one deliberately selected frame.
* Confidence: High.

### 3. Manual attendance endpoints bypass shift windows

* File: `src/modules/attendance/api.py:17`, `src/modules/attendance/service.py:123`
* Problem: REST check-in/out calls `check_in` and `check_out` directly. Only recognition calls `log_attendance`, which checks configured shift windows.
* Impact: Manual calls can record attendance at any time, contradicting API documentation and business rules.
* Suggested fix: Route all attendance decisions through one policy function, or make an explicitly authorized override endpoint with audit logging.
* Confidence: High.

### 4. Recognition backpressure is ineffective

* File: `src/modules/recognition/pipeline.py:27`, `src/platform/queue.py:12`
* Problem: The pipeline drains the bounded queue into an unbounded set of tasks. Only four executor workers perform CPU work, so pending tasks and frame bytes can grow without bound. One high-rate client can also dominate the global queue.
* Impact: Memory exhaustion, stale-frame processing, unfair service across cameras, and denial of service.
* Suggested fix: Use a fixed worker pool or semaphore before dequeuing. Prefer a bounded latest-frame slot per client and enforce frame-rate/size limits.
* Confidence: High.

### 5. PostgreSQL and Qdrant updates are not recoverably coordinated

* File: `src/modules/employees/service.py:42`
* Problem: Registration commits PostgreSQL, deletes existing vectors, then upserts Qdrant. Removal deletes Qdrant first, then commits PostgreSQL. Failures leave missing vectors, orphaned employees, or deleted recognition data.
* Impact: Employees can become unrecognizable or retain inconsistent identity data after partial failures.
* Suggested fix: Introduce an outbox/reconciliation workflow with idempotent vector operations and observable retry state.
* Confidence: High.

### 6. Employee registration silently overwrites existing identities

* File: `src/modules/employees/service.py:23`
* Problem: `POST /employees` behaves as an upsert: an existing `emp_code` has its name and all embeddings replaced.
* Impact: Accidental code reuse or unauthorized requests can take over an existing employee’s biometric identity.
* Suggested fix: Make create return HTTP 409 for duplicate codes. Provide a distinct, authorized re-enrollment operation with explicit confirmation and audit history.
* Confidence: High.

### 7. Deleting an employee cascades through attendance history

* File: `initdb/init.sql:14`, `src/modules/employees/models.py:14`
* Problem: Both database and ORM relationships delete attendance records with the employee.
* Impact: Historical payroll/audit evidence can be erased through employee deletion.
* Suggested fix: Verify retention requirements. Normally use soft-deactivation, preserve immutable attendance records, and restrict hard deletion.
* Confidence: High technically; retention policy needs verification.

### 8. Database/vector services are directly exposed and Qdrant is unpinned

* File: `compose.yaml:1`
* Problem: PostgreSQL and unauthenticated Qdrant ports are published to the host. Qdrant uses the mutable `latest` tag.
* Impact: Network-accessible biometric vectors or database attacks; unreviewed upgrades can break or alter production behavior.
* Suggested fix: Keep data services on an internal Compose network, enable Qdrant authentication/TLS where appropriate, and pin images by version or digest.
* Confidence: High for configuration; actual reachability needs verification.

## Medium Priority Issues

### 1. App factory mutates a global router

* File: `src/modules/recognition/ws_ingress.py:8`
* Problem: Each `create_app()` call adds another endpoint to the same module-global router. Verified route counts were `1` then `2`.
* Impact: Duplicate routes and retained closures in tests, reloaders, or multi-app processes.
* Suggested fix: Construct a new `APIRouter` inside `make_ws_router`.
* Confidence: High.

### 2. Internal exception details are returned to clients

* File: `src/modules/employees/service.py:59`, `src/modules/attendance/service.py:38`, `src/modules/recognition/pipeline.py:74`
* Problem: Raw database/Qdrant exception strings become HTTP or WebSocket responses.
* Impact: Information disclosure and unstable client-facing error contracts.
* Suggested fix: Log structured internal errors server-side and return stable, generic error codes.
* Confidence: High.

### 3. No schema migration mechanism

* File: `initdb/init.sql:1`, `src/modules/attendance/models.py:7`
* Problem: Schema exists in raw initialization SQL and separate ORM declarations, but no migration tooling exists.
* Impact: Existing volumes cannot safely receive schema changes; SQL constraints and ORM models can drift.
* Suggested fix: Adopt versioned migrations and test both fresh installation and upgrades.
* Confidence: High.

### 4. Lifecycle cleanup is incomplete

* File: `src/app.py:25`
* Problem: Shutdown cancels the pipeline without awaiting it and does not close Qdrant, dispose the SQLAlchemy engine, or shut down executors.
* Impact: In-flight work may be abandoned and reload/test processes can leak resources.
* Suggested fix: Cancel and await tasks under `CancelledError`, then close clients, engines, and executors deterministically.
* Confidence: High.

### 5. Time and shift invariants are weak

* File: `src/modules/attendance/schemas.py:5`, `src/modules/attendance/service.py:60`
* Problem: Shift ranges can overlap, date filters allow `from_date > to_date`, and attendance uses naive host-local timestamps while WebSocket responses use UTC.
* Impact: Ambiguous check-in/out selection and deployment-dependent dates.
* Suggested fix: Validate shift/date relationships and define one explicit business timezone with timezone-aware storage.
* Confidence: High.

### 6. Current dependency audit is not clean

* File: `frontend/package-lock.json:1`
* Problem: `npm audit --omit=dev` reports high advisories in `form-data` through Axios and `hono` through `shadcn`, plus a moderate `js-yaml` advisory.
* Impact: Supply-chain risk; application reachability differs by package.
* Suggested fix: Update the lockfile after assessing reachability. Move CLI/build-only packages out of production dependencies where possible.
* Confidence: High audit result; runtime reachability needs verification.

### 7. Deployment and documentation contracts have drifted

* File: `docs/setup/getting-started.md:8`, `docs/api/api_spec.md:118`, `README.md:15`
* Problem: Documentation says Node 18+, but installed Next requires Node `>=20.9.0`; API docs still show the removed attendance request body; README describes pgvector although the code uses Qdrant; `/kiosk` is documented but absent.
* Impact: Failed setup, incorrect client implementations, and misleading architecture decisions.
* Suggested fix: Make generated OpenAPI and current runtime requirements authoritative; remove or archive obsolete architecture documents.
* Confidence: High.

### 8. No CI or production deployment quality gates

* File: `pyproject.toml:1`, `frontend/package.json:5`
* Problem: No CI workflow, backend lint/type configuration, frontend test script, Dockerfile, health checks, or production server command is present. `main.py` always enables reload when run directly.
* Impact: Regressions and incompatible environments can ship without detection.
* Suggested fix: Add CI for tests, lint, typecheck, dependency audit, migration verification, and production builds; define explicit dev and production commands.
* Confidence: High.

### 9. Frontend camera processing wastes resources

* File: `frontend/src/components/attendance/attendance-client.tsx:47`, `frontend/src/components/employees/employee-registration.tsx:249`
* Problem: Attendance repeatedly creates full JPEG data URLs that are not sent or rendered. Registration creates object URLs during render and never revokes them.
* Impact: Avoidable CPU and memory usage during long kiosk/enrollment sessions.
* Suggested fix: Remove unused snapshots and create/revoke preview URLs in controlled lifecycle hooks.
* Confidence: High.

## Low Priority / Cleanup

### 1. Numeric employee names cannot be searched

* File: `src/modules/employees/api.py:71`
* Problem: Any numeric path segment is interpreted as an internal ID.
* Suggested fix: Use separate `/employees/{id}` and `/employees/search?name=` routes.

### 2. Unbounded employee-name search

* File: `src/modules/employees/service.py:122`
* Problem: Search returns all matches without pagination.
* Suggested fix: Add a bounded limit and pagination.

### 3. Compatibility alias exists only for tests

* File: `src/modules/recognition/identifier.py:27`
* Problem: Production uses `identify_face`; the `identify` alias is retained solely for tests/legacy planning.
* Suggested fix: Update tests to the canonical name and remove the alias when compatibility is no longer required.

## Unnecessary Abstractions

* `frontend/src/components/theme-provider.tsx:1`: 180-line reimplementation of functionality already provided by the installed `next-themes` dependency.
* `frontend/src/lib/api.ts:103`: accepts multiple speculative response shapes (`items`, `data`, `results`, raw arrays) despite one controlled backend contract. This hides drift instead of detecting it.
* `src/modules/employees/service.py:15`: pervasive `(value, error_string)` returns add branching and encourage exception leakage. Typed domain exceptions plus centralized API mapping would be clearer.
* `frontend/src/lib/shift-settings.tsx:42`: manually duplicates server-state loading/caching while React Query is already globally installed. Preserve it only if offline local fallback is an explicit requirement.
* `src/modules/recognition/ws_ingress.py:11`: router factory around a global mutable router adds indirection and causes route duplication.

## Dead Code / Unused Code

* `src/platform/db/session.py:35`: `get_connection()` has no production caller; it keeps synchronous psycopg2 in an otherwise async backend.
* `src/platform/config.py:10`: `ORIGINAL_IMG_PATH` and `MAX_EMB_FACE` are unused.
* `frontend/src/lib/api.ts:183`: `listAttendanceHistory()` has no caller.
* `frontend/src/lib/mock-data.ts:60`: `recentAttendance` has no caller.
* `frontend/src/lib/format.ts:5`: `formatPercent` is unused.
* `frontend/src/components/ui/textarea.tsx:1`: component has no consumer.
* `src/platform/queue.py:9`: `captured_at` is populated but never used for staleness or ordering.
* `frontend/public/*.svg` and `template/face_icon.png` appear unused.

## Unused or Suspicious Dependencies

* `plotly`: no source import; remove if no external script depends on it.
* `psycopg2-binary`: used only by the apparently dead synchronous `get_connection`.
* `next-themes`: unused because of the custom theme implementation.
* `shadcn`: CLI/build tooling installed as a production dependency; it also pulls the audited Hono chain. Its CSS import means removal needs build verification.
* `fastapi[standard]` plus direct `uvicorn`: partly redundant, although an explicit Uvicorn pin can be intentional.
* `qdrant/qdrant:latest`: mutable container dependency, unsuitable for reproducible deployment.

## Duplicated Logic

* Shift defaults exist independently in `initdb/init.sql:30`, `src/modules/attendance/service.py:7`, and `frontend/src/lib/shift-settings.tsx:26`. Make persisted server settings canonical.
* Month labels are duplicated in `mock-data.ts` and `dashboard-client.tsx`.
* Recognition and enrollment maintain separate four-worker executors while sharing one InsightFace singleton. Consolidation could reduce GPU/CPU oversubscription; thread safety needs verification.
* Several frontend API functions repeat permissive `items/data/results` normalization. If compatibility is genuinely needed, use one tested boundary decoder.
* Architecture, API, setup, and planning documents repeat contracts that have already diverged.

## Security Review

* No authentication, authorization, administrator roles, kiosk credentials, or audit trail.
* Destructive employee removal also deletes attendance history.
* Biometric vectors and identifying payloads are stored in unauthenticated Qdrant.
* Pass-through liveness permits presentation attacks.
* Enrollment accepts multiple faces under one identity.
* Upload and WebSocket frame sizes/rates are unrestricted.
* Reusing a `client_id` replaces another WebSocket connection; disconnecting the old connection can remove the new mapping.
* Raw internal errors are disclosed.
* No explicit CSP/HSTS/frame/security-header configuration.
* No request-rate limiting.
* Qdrant and PostgreSQL are host-published.
* npm audit reports unresolved advisories.
* Database and vector transport/encryption-at-rest requirements need verification.

## Performance Review

* Recognition tasks accumulate without a concurrency bound.
* One global queue does not provide per-camera fairness.
* Stale frames are not discarded by age.
* Two independent executors can concurrently invoke the same model singleton; GPU behavior needs verification.
* Uploads have no byte or decoded-pixel limits.
* Employee-name search is unbounded.
* Attendance UI performs unused screenshot encoding.
* Registration preview URLs leak.
* No performance budgets, profiling evidence, production metrics, or queue-depth telemetry exist.

## Testing Gaps

* No backend API integration tests; these would have caught the JSON-body/query mismatch.
* No frontend component, API-contract, or end-to-end tests.
* No WebSocket tests for authentication, duplicate IDs, disconnects, frame limits, routing, or fairness.
* No pipeline concurrency/backpressure/stale-frame tests.
* No real PostgreSQL tests for unique constraints, concurrent check-in, generated duration, cascade behavior, or migrations.
* No Qdrant/PostgreSQL partial-failure and reconciliation tests.
* No upload tests for oversized files, corrupt images, decompression bombs, zero faces, or multiple faces.
* No authorization or security-header tests.
* No liveness/spoofing acceptance tests.
* No app lifecycle/shutdown tests.
* No clean-install or upgrade/deployment verification.

## Suggested Refactor Plan

1. Quick wins
   1. Make `db-down` non-destructive.
   2. Remove fake detection or clearly isolate it as demo-only.
   3. Align frontend requests with generated OpenAPI.
   4. Add upload/frame limits and exactly-one-face validation.
   5. Stop returning internal exception strings.
   6. Fix the global router factory.
   7. Remove confirmed dead exports/dependencies and update documentation.

2. Medium refactors
   1. Add admin/kiosk authentication, authorization, and audit events.
   2. Introduce bounded, per-client recognition workers.
   3. Add migrations and explicit timezone/shift validation.
   4. Replace custom server-state/theme abstractions with existing maintained libraries where appropriate.
   5. Add API, WebSocket, database, and frontend integration tests plus CI gates.

3. Risky/large refactors
   1. Implement outbox/reconciliation for PostgreSQL and Qdrant.
   2. Establish employee deactivation and attendance-retention policy.
   3. Implement and calibrate real liveness detection.
   4. Harden deployment networking, secrets, TLS, observability, and recovery procedures.

## Do Not Change

* The `platform/` versus domain `modules/` boundary is understandable and appropriately small.
* SQLAlchemy queries are parameterized; no SQL-string injection pattern was found.
* List/history endpoints already cap page size at 100.
* The partial unique index preventing multiple open logs per employee/day is valuable.
* The database checkout-after-checkin constraint should remain.
* The Qdrant threshold conversion is consistent with the documented legacy distance semantics; do not change it without recalibration data.
* The drop-oldest realtime policy is reasonable, but it needs bounded workers and per-client fairness.
* Keep the `LivenessChecker` seam; replace the pass-through implementation rather than removing the interface.
* Generated UI primitives appear stable; avoid refactoring them without a functional requirement.
