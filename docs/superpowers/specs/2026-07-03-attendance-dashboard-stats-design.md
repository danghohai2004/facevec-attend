# Attendance Dashboard Stats API — Design

## Problem

The frontend Dashboard (`frontend/src/components/dashboard/dashboard-client.tsx`) renders summary
cards and two charts entirely from hardcoded data in `frontend/src/lib/mock-data.ts`
(`summaryMetrics`, `getMonthlyStats`, `getDailyAverages`, `availableYears`). The backend has no
endpoint that returns aggregated attendance statistics — `GET /api/attendance` only returns a
paginated history for a single `emp_id`. This design adds the backend endpoints needed to replace
the mock data with real data, without changing the Dashboard's UI or chart shape.

## Non-goals

- No changes to the Dashboard's React components, charts, or layout — this is backend-only.
- No changes to `Employee` or `AttendanceLog` table schemas.
- No auth changes beyond following the existing pattern (see below).

## Endpoints

All three are added to the existing `src/modules/attendance/api.py` router
(`APIRouter(prefix="/api", tags=["Attendance"])`), so full paths are `/api/attendance/summary`,
`/api/attendance/monthly`, `/api/attendance/daily`. None require `require_api_key` — consistent
with the existing `GET /api/attendance` and `GET /api/employees` (only write operations are
protected in this codebase).

### 1. `GET /api/attendance/summary`

No query parameters. Always computed relative to "today" in `BUSINESS_TIMEZONE`
(`Asia/Ho_Chi_Minh`, already defined in `src/modules/attendance/service.py`).

Response (`SummaryStatsResponse`):
```json
{
  "total_employees": 128,
  "todays_attendance": 94,
  "average_working_hours": 7.6,
  "on_time_rate": 91.5,
  "deltas": {
    "todays_attendance": 2.1,
    "average_working_hours": -1.3,
    "on_time_rate": 3.4
  }
}
```

Field definitions:
- `total_employees`: `COUNT(*)` over `employees`. **No delta** — the `employees` table has no
  `created_at` column, so day-over-day headcount change cannot be computed. This was confirmed
  with the user; do not add a migration for this.
- `todays_attendance`: `COUNT(*)` of `attendance_logs` where `working_date = today`.
- `average_working_hours`: `AVG(working_duration)` in hours, over today's logs where
  `working_duration IS NOT NULL` (i.e. checked out). If there are zero such logs, return `0.0`.
- `on_time_rate`: `100 * (count of today's logs where checkin_time.time() <= shift.check_in_end) /
  todays_attendance`. If `todays_attendance == 0`, return `0.0`. Uses the single current
  `ShiftSettings` row (via `get_shift_settings`, same helper `manual_check_in` uses); if none
  exists, fall back to `_DEFAULT_SHIFT` exactly like the rest of the module already does.
- `deltas.*`: percent change of that metric vs. the same metric computed for **yesterday**
  (`(today_value - yesterday_value) / yesterday_value * 100`). If the yesterday value is `0`,
  return `null` for that delta (avoid divide-by-zero / misleading infinite deltas).

### 2. `GET /api/attendance/monthly?year=<int>`

`year` is required, `int`, no explicit range validation (a nonexistent year simply returns an
empty `items` list — not an error).

Response (`MonthlyStatsResponse`):
```json
{
  "available_years": [2024, 2025],
  "items": [
    { "month": 1, "attendance": 2098, "working_hours": 15945.0, "average_hours": 7.4 }
  ]
}
```

- `available_years`: `SELECT DISTINCT EXTRACT(YEAR FROM working_date) FROM attendance_logs ORDER
  BY 1`, cast to `int`, computed unconditionally on every call to this endpoint (cheap distinct
  scan; no separate endpoint needed). If the table is empty, returns `[]`.
- `items`: one entry **per month that has at least one log** in the requested year (`GROUP BY
  EXTRACT(MONTH FROM working_date)`), months with zero logs are omitted (frontend already handles
  sparse arrays via `Recharts`, which just won't plot a missing category — matches how `daily`
  below is already sparse).
  - `attendance` = `COUNT(*)` logs in that month.
  - `working_hours` = `SUM(working_duration)` in hours, over logs where `working_duration IS NOT
    NULL`.
  - `average_hours` = `AVG(working_duration)` in hours, same filter. Round to 1 decimal (matches
    mock precision) — do the rounding in Python after the query, not in SQL.

### 3. `GET /api/attendance/daily?year=<int>&month=<int, 1-12>`

Both required. `month` outside 1-12 → `422` (FastAPI `Query(..., ge=1, le=12)` handles this for
free, same pattern as `page_size: int = Query(20, ge=1, le=100)` elsewhere in this module).

Response (`DailyStatsResponse`):
```json
{
  "items": [
    { "day": 1, "average_hours": 7.4 }
  ]
}
```

- One entry **per day that has at least one log** in that year+month (`GROUP BY
  EXTRACT(DAY FROM working_date)`). Days with no logs are omitted — do **not** synthesize a full
  28-31 day array of zeros (the current mock does this arbitrarily; real data should only report
  days that actually happened).
- `average_hours` = `AVG(working_duration)` in hours over logs where `working_duration IS NOT
  NULL`, rounded to 2 decimals in Python (matches mock precision).

## Data notes (apply to all three endpoints)

- `working_duration` is a Postgres-generated `Interval` column (see
  `src/modules/attendance/models.py`); convert to hours via
  `func.extract("epoch", AttendanceLog.working_duration) / 3600.0` in the SQLAlchemy query, then
  alias it (e.g. `.label("hours")`) so `AVG`/`SUM` operate on the numeric value.
- A log with `checkout_time IS NULL` (still checked in) has `working_duration IS NULL` — it counts
  toward attendance/count metrics but is excluded from any `AVG`/`SUM` of hours (SQL `AVG`/`SUM`
  already skip `NULL` by default, so no extra `WHERE` clause is needed beyond what the aggregate
  does naturally — just don't `COALESCE` it to 0).
- "Today" and "yesterday" for the summary endpoint must be computed via the existing
  `BUSINESS_TIMEZONE` (`Asia/Ho_Chi_Minh`), the same timezone `manual_check_in`/`manual_check_out`
  already use — do not use naive UTC `date.today()`.
- On-time comparison (`checkin_time.time() <= shift.check_in_end`) should be done as a SQL
  `WHERE`/`CASE` condition (comparing the `Time` column extracted from `checkin_time` against
  `shift.check_in_end`), not pulled into Python row-by-row — keep it a single aggregate query per
  endpoint. `_is_time_in_range` in `service.py` handles overnight-shift wraparound for the
  check-in/check-out *window* test; on-time comparison here is a simpler one-sided `<=` against
  `check_in_end` and does not need that helper.

## Files touched

- `src/modules/attendance/schemas.py` — add `SummaryStatsResponse`, `SummaryDeltas`,
  `MonthlyStatsResponse`, `MonthlyStatItem`, `DailyStatsResponse`, `DailyStatItem`.
- `src/modules/attendance/service.py` — add `get_summary_stats`, `get_monthly_stats`,
  `get_daily_stats` (async, `db: AsyncSession` first arg, follow the existing
  `tuple[T, str | None]` error-return convention used by every other function in this file).
- `src/modules/attendance/api.py` — add the three `@router.get` routes, no
  `dependencies=[Depends(require_api_key)]`, following the existing `GET /api/attendance` route's
  error-handling shape (`if err: raise HTTPException(500, err)`).
- `tests/modules/attendance/` — add/extend tests for the three new service functions (see
  Testing below). Check whether `tests/modules/attendance/` already exists before assuming its
  layout; mirror `tests/modules/recognition/test_identifier.py`'s style if a sibling test file for
  this module doesn't already exist.

## Testing

Follow whatever test framework/fixtures the existing `tests/` directory already uses for DB-backed
tests (inspect `tests/modules/attendance/` or `tests/modules/employees/` for a fixture pattern —
likely a test DB session fixture, given `AsyncSession` is used throughout). At minimum:

- `get_summary_stats`: seed a few `attendance_logs` rows for today and yesterday with varying
  `checkin_time`/`checkout_time`, assert counts, average hours, on-time rate, and delta math
  (including the "yesterday value is 0 → delta is `null`" case).
- `get_monthly_stats`: seed logs across 2+ months and 2+ years, assert `available_years` is
  correct and `items` only contains months with data.
- `get_daily_stats`: seed logs across 2+ days in one month, assert `items` only contains days with
  data, and that a log with `checkout_time IS NULL` is excluded from `average_hours` but doesn't
  break the query.
- One test per endpoint (`api.py`) confirming the route wires to the service function and returns
  the right HTTP shape — doesn't need to duplicate all the service-level edge cases.

## Explicitly out of scope / deferred

- Frontend integration (swapping `mock-data.ts` for real `fetch` calls in
  `dashboard-client.tsx`) — separate follow-up task once this API exists.
- `total_employees` delta — deferred; would require a schema migration (`created_at` on
  `Employee`), which the user declined for this iteration.
- Caching/memoization of these aggregate queries — not needed at current expected data volume;
  revisit if the `attendance_logs` table grows large enough that repeated `COUNT`/`AVG` scans on
  every dashboard load become slow.
