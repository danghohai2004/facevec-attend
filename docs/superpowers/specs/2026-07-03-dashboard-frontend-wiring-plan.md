# Plan: Wire Dashboard frontend to the real attendance-stats API

## Context

The backend now has 3 working, reviewed, tested endpoints (implemented per
`docs/superpowers/specs/2026-07-03-attendance-dashboard-stats-design.md`) that the frontend
Dashboard does not call yet. `frontend/src/components/dashboard/dashboard-client.tsx` still
imports everything from `frontend/src/lib/mock-data.ts` (hardcoded numbers). This plan replaces
that with real `fetch`/`react-query` calls to the new endpoints. No backend changes are needed —
this is 100% frontend.

This plan is self-contained for an implementer with no access to prior conversation — every step
has exact file paths and exact before/after code.

## Repo root
`/home/dhhai/Workspace/facevec-attend`

## Backend contract (already implemented and tested — do not modify backend)

Base URL is `frontend/src/lib/api.ts`'s `api` axios instance (`baseURL` = `NEXT_PUBLIC_API_BASE_URL`
+ `/api`, no auth needed for GET — these are public reads, same as `getShiftSettings`).

### `GET /attendance/summary` — no params
```ts
{
  total_employees: number;
  todays_attendance: number;
  average_working_hours: number;
  on_time_rate: number;
  deltas: {
    todays_attendance: number | null;
    average_working_hours: number | null;
    on_time_rate: number | null;
  };
}
```
Note: `total_employees` has **no** delta field at all (backend intentionally omits it — see design
spec). There is no `deltas.total_employees`.

### `GET /attendance/monthly?year=<int>`
```ts
{
  available_years: number[];
  items: Array<{ month: number; attendance: number; working_hours: number; average_hours: number }>;
}
```
`items` only contains months that have data (sparse — a year with only February and March logged
returns 2 items, not 12). `month` is `1`-`12` (not a name).

### `GET /attendance/daily?year=<int>&month=<int 1-12>`
```ts
{
  items: Array<{ day: number; average_hours: number }>;
}
```
`items` only contains days that have data (sparse). `day` is `1`-`31` (not zero-padded string).

## Step 1 — Add types

**File:** `frontend/src/lib/types.ts`

Append at the end of the file (after the existing `EmployeeList` type):

```ts
export type SummaryDeltas = {
  todaysAttendance: number | null;
  averageWorkingHours: number | null;
  onTimeRate: number | null;
};

export type SummaryStats = {
  totalEmployees: number;
  todaysAttendance: number;
  averageWorkingHours: number;
  onTimeRate: number;
  deltas: SummaryDeltas;
};

export type MonthlyStat = {
  month: number;
  attendance: number;
  workingHours: number;
  averageHours: number;
};

export type MonthlyStats = {
  availableYears: number[];
  items: MonthlyStat[];
};

export type DailyStat = {
  day: number;
  averageHours: number;
};

export type DailyStats = {
  items: DailyStat[];
};
```

(These are camelCase frontend-facing types; the API functions in Step 2 convert the backend's
snake_case JSON into these shapes, following the same convention `normalizeEmployee` and
`normalizeShiftSettings` already use in `api.ts`.)

## Step 2 — Add API functions

**File:** `frontend/src/lib/api.ts`

Update the type import at the top of the file:

```ts
import type {
  DailyStats,
  Employee,
  EmployeeList,
  MonthlyStats,
  ShiftSettings,
  SummaryStats,
} from "@/lib/types";
```

Append these three functions at the end of the file (after `updateShiftSettings`):

```ts
export async function getSummaryStats(): Promise<SummaryStats> {
  const response = await api.get("/attendance/summary");
  const data = response.data;
  return {
    totalEmployees: data.total_employees,
    todaysAttendance: data.todays_attendance,
    averageWorkingHours: data.average_working_hours,
    onTimeRate: data.on_time_rate,
    deltas: {
      todaysAttendance: data.deltas.todays_attendance,
      averageWorkingHours: data.deltas.average_working_hours,
      onTimeRate: data.deltas.on_time_rate,
    },
  };
}

export async function getMonthlyStats(year: number): Promise<MonthlyStats> {
  const response = await api.get("/attendance/monthly", { params: { year } });
  const data = response.data;
  return {
    availableYears: data.available_years,
    items: data.items.map((item: Record<string, number>) => ({
      month: item.month,
      attendance: item.attendance,
      workingHours: item.working_hours,
      averageHours: item.average_hours,
    })),
  };
}

export async function getDailyStats(year: number, month: number): Promise<DailyStats> {
  const response = await api.get("/attendance/daily", { params: { year, month } });
  const data = response.data;
  return {
    items: data.items.map((item: Record<string, number>) => ({
      day: item.day,
      averageHours: item.average_hours,
    })),
  };
}
```

No try/catch or 404-fallback needed here (unlike `getEmployeesByName`) — these endpoints always
return `200` with a (possibly empty) `items`/sparse response, they don't 404.

## Step 3 — Rewrite `dashboard-client.tsx` to fetch real data

**File:** `frontend/src/components/dashboard/dashboard-client.tsx`

This file already uses `@tanstack/react-query` elsewhere in the codebase (see
`frontend/src/components/employees/employee-list.tsx` for the established pattern:
`useQuery({ queryKey: [...], queryFn: () => ... })`). Follow that same pattern here — this file
does NOT currently import `useQuery`, so it must be added.

### 3a. Replace the mock-data import and add real imports

Find:
```tsx
import {
  availableYears,
  getDailyAverages,
  getMonthlyStats,
  summaryMetrics,
} from "@/lib/mock-data";
import { formatHours } from "@/lib/format";
import { cn } from "@/lib/utils";
```

Replace with:
```tsx
import { useQuery } from "@tanstack/react-query";
import { getDailyStats, getMonthlyStats, getSummaryStats } from "@/lib/api";
import { formatHours } from "@/lib/format";
import { cn } from "@/lib/utils";
```

(Note the name collision: the mock module exported a function also called `getMonthlyStats`. The
new import from `@/lib/api` replaces it — after this edit there is only one `getMonthlyStats` in
scope, from `@/lib/api`.)

### 3b. Replace state/data-fetching logic inside `DashboardClient`

Find:
```tsx
export function DashboardClient() {
  const currentYear = new Date().getFullYear();
  const defaultYear = availableYears.includes(currentYear)
    ? currentYear
    : availableYears[availableYears.length - 1];
  const defaultMonth = monthOptions[new Date().getMonth()];

  const [selectedYear, setSelectedYear] = React.useState(`${defaultYear}`);
  const [selectedMonth, setSelectedMonth] = React.useState(defaultMonth);

  const monthlyStats = React.useMemo(
    () => getMonthlyStats(Number(selectedYear)),
    [selectedYear],
  );

  const dailyAverages = React.useMemo(
    () => getDailyAverages(Number(selectedYear), selectedMonth),
    [selectedYear, selectedMonth],
  );

  const { deltas } = summaryMetrics;
```

Replace with:
```tsx
export function DashboardClient() {
  const currentYear = new Date().getFullYear();
  const defaultMonth = monthOptions[new Date().getMonth()];

  const [selectedYear, setSelectedYear] = React.useState(`${currentYear}`);
  const [selectedMonth, setSelectedMonth] = React.useState(defaultMonth);

  const summaryQuery = useQuery({
    queryKey: ["attendance-summary"],
    queryFn: getSummaryStats,
  });

  const monthlyQuery = useQuery({
    queryKey: ["attendance-monthly", selectedYear],
    queryFn: () => getMonthlyStats(Number(selectedYear)),
  });

  const dailyQuery = useQuery({
    queryKey: ["attendance-daily", selectedYear, selectedMonth],
    queryFn: () =>
      getDailyStats(Number(selectedYear), monthOptions.indexOf(selectedMonth) + 1),
  });

  const summary = summaryQuery.data;
  const deltas = summary?.deltas;
  const availableYears = monthlyQuery.data?.availableYears ?? [currentYear];
  const monthlyStats = monthlyQuery.data?.items ?? [];
  const dailyAverages = dailyQuery.data?.items ?? [];
```

Why the default-year logic changed: the old mock had a hardcoded `availableYears` array known
upfront, so it could pick a sensible default before any data loaded. The real `available_years`
list only arrives after `monthlyQuery` resolves, so the year `<Select>` must default to the
current calendar year (matches how `shift-settings-form.tsx` and other forms in this codebase
default to "now" before data loads) and simply re-render with the real list once the query
resolves — no need to block rendering on it.

### 3c. Update the 4 `StatCard`s to read from `summary` (with `summaryQuery.isLoading` handling)

Find:
```tsx
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          title="Total Employees"
          value={`${summaryMetrics.totalEmployees}`}
          delta={deltas.totalEmployees}
          hint="active on system"
          icon={Users}
          accent="yellow"
        />
        <StatCard
          title="Today's Attendance"
          value={`${summaryMetrics.todaysAttendance}`}
          delta={deltas.todaysAttendance}
          hint="check-ins so far"
          icon={UserCheck}
          accent="cyan"
        />
        <StatCard
          title="Average Working Hours"
          value={formatHours(summaryMetrics.averageWorkingHours)}
          delta={deltas.averageWorkingHours}
          hint="last 30 days"
          icon={Clock}
          accent="pink"
        />
        <StatCard
          title="On-Time Rate"
          value={`${summaryMetrics.onTimeRate}%`}
          delta={deltas.onTimeRate}
          hint="vs. last month"
          icon={TrendingUp}
          accent="lime"
        />
      </div>
```

Replace with:
```tsx
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          title="Total Employees"
          value={summary ? `${summary.totalEmployees}` : "—"}
          delta={null}
          hint="active on system"
          icon={Users}
          accent="yellow"
        />
        <StatCard
          title="Today's Attendance"
          value={summary ? `${summary.todaysAttendance}` : "—"}
          delta={deltas?.todaysAttendance ?? null}
          hint="check-ins so far"
          icon={UserCheck}
          accent="cyan"
        />
        <StatCard
          title="Average Working Hours"
          value={summary ? formatHours(summary.averageWorkingHours) : "—"}
          delta={deltas?.averageWorkingHours ?? null}
          hint="today"
          icon={Clock}
          accent="pink"
        />
        <StatCard
          title="On-Time Rate"
          value={summary ? `${summary.onTimeRate}%` : "—"}
          delta={deltas?.onTimeRate ?? null}
          hint="today"
          icon={TrendingUp}
          accent="lime"
        />
      </div>
```

Two things changed on purpose here, matching what the backend actually computes (per the design
spec) rather than what the old mock pretended:
- `Total Employees` now always passes `delta={null}` — the backend has no
  `deltas.total_employees` field at all (no `created_at` column to compute it from). The old mock
  faked a delta for this card; the real card must not.
- The `hint` text for "Average Working Hours" and "On-Time Rate" changed from `"last 30 days"` /
  `"vs. last month"` to `"today"`, because the backend's `deltas` are day-over-day (today vs.
  yesterday), not month-over-month — the old hint text was describing the mock's fake period, not
  the real one. Keep this change; don't try to preserve the old wording.

### 3d. Update `StatCard`'s `delta` prop to accept `null`

Find:
```tsx
function StatCard({
  title,
  value,
  delta,
  hint,
  icon: Icon,
  accent,
}: {
  title: string;
  value: string;
  delta: number;
  hint: string;
  icon: LucideIcon;
  accent: PosterAccent;
}) {
  const up = delta >= 0;
  return (
    <Card className={cn(posterBg[accent], "border-ink text-ink shadow-poster")}>
      <CardHeader className="flex-row items-center justify-between gap-2 space-y-0">
        <CardTitle className="text-xs sm:text-sm">{title}</CardTitle>
        <span className="flex h-9 w-9 items-center justify-center rounded-[3px] border-2 border-ink bg-white">
          <Icon className="h-4 w-4" />
        </span>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="text-4xl font-black tracking-tight tabular-nums sm:text-5xl">
          {value}
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span
            className={cn(
              "inline-flex items-center gap-0.5 rounded-[3px] border-2 border-ink bg-white px-1.5 py-0.5 font-bold tabular-nums",
              up ? "text-emerald-700" : "text-rose-700",
            )}
          >
            {up ? (
              <ArrowUpRight className="h-3 w-3" />
            ) : (
              <ArrowDownRight className="h-3 w-3" />
            )}
            {Math.abs(delta).toFixed(1)}%
          </span>
          <span className="font-bold text-ink/70">{hint}</span>
        </div>
      </CardContent>
    </Card>
  );
}
```

Replace with:
```tsx
function StatCard({
  title,
  value,
  delta,
  hint,
  icon: Icon,
  accent,
}: {
  title: string;
  value: string;
  delta: number | null;
  hint: string;
  icon: LucideIcon;
  accent: PosterAccent;
}) {
  const up = delta !== null && delta >= 0;
  return (
    <Card className={cn(posterBg[accent], "border-ink text-ink shadow-poster")}>
      <CardHeader className="flex-row items-center justify-between gap-2 space-y-0">
        <CardTitle className="text-xs sm:text-sm">{title}</CardTitle>
        <span className="flex h-9 w-9 items-center justify-center rounded-[3px] border-2 border-ink bg-white">
          <Icon className="h-4 w-4" />
        </span>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="text-4xl font-black tracking-tight tabular-nums sm:text-5xl">
          {value}
        </div>
        <div className="flex items-center gap-2 text-xs">
          {delta !== null && (
            <span
              className={cn(
                "inline-flex items-center gap-0.5 rounded-[3px] border-2 border-ink bg-white px-1.5 py-0.5 font-bold tabular-nums",
                up ? "text-emerald-700" : "text-rose-700",
              )}
            >
              {up ? (
                <ArrowUpRight className="h-3 w-3" />
              ) : (
                <ArrowDownRight className="h-3 w-3" />
              )}
              {Math.abs(delta).toFixed(1)}%
            </span>
          )}
          <span className="font-bold text-ink/70">{hint}</span>
        </div>
      </CardContent>
    </Card>
  );
}
```

`delta === null` covers both cases that now legitimately happen with real data: `Total Employees`
(no delta field exists at all) and any metric whose `deltas.*` came back `null` from the backend
(yesterday's value was `0`, so percent-change is undefined — see design spec). In both cases the
badge is hidden instead of rendering `NaN%` or a misleading `0.0%`.

## Step 4 — Delete the mock data file

**File:** `frontend/src/lib/mock-data.ts`

Delete this file entirely (`git rm frontend/src/lib/mock-data.ts` or just remove it) — after Step
3, nothing in the codebase imports from it (`demoEmployees` in this file is also unused; confirm
with `grep -rn "mock-data\|demoEmployees" frontend/src` before deleting — it should return zero
matches once Step 3 is done and this file is removed).

## Verification

1. `cd frontend && npx tsc --noEmit` — must pass with zero errors (confirms the type changes in
   Step 1/2 and the `StatCard` signature change in Step 3d are all consistent).
2. `cd frontend && npx eslint src/components/dashboard/dashboard-client.tsx src/lib/api.ts src/lib/types.ts`
   — must pass.
3. Start the backend (`uv run uvicorn src.app:app --reload` or whatever the project's existing dev
   command is — check `README.md`) and the frontend dev server (`npm run dev` in `frontend/`).
4. Open the Dashboard page in a browser:
   - The 4 summary cards show real numbers from the database (not `128`, `94`, `7.6`, `91.5` —
     those were the old mock's hardcoded values, so seeing different numbers confirms it's wired
     correctly; seeing exactly those numbers again would be a red flag).
   - "Total Employees" card shows no delta badge (no up/down arrow).
   - Changing the Year `<Select>` triggers a new `/attendance/monthly` request (visible in the
     browser Network tab) and updates both the "Monthly Attendance Trend" and "Total Working
     Hours" charts.
   - Changing the Month `<Select>` triggers a new `/attendance/daily` request and updates the
     "Average Hours per Day" chart.
   - If the database has no attendance logs at all, the charts render empty (no crash, no
     `undefined` errors in the console) — this is the sparse-data case the backend intentionally
     returns.
5. Confirm `git grep -n "mock-data"` in `frontend/src` returns nothing after Step 4.
