# Monthly Attendance Report Export (Excel)

Date: 2026-07-03
Status: approved

## Goal

HR can download a monthly attendance report as an `.xlsx` file from the dashboard.

## Backend

New endpoint in the existing attendance module:

- `GET /api/attendance/report?year=<int>&month=<1-12>`
- Response: `attendance_YYYY-MM.xlsx` via streaming response with
  `Content-Disposition: attachment`.
- New dependency: `openpyxl`.

Workbook, built in-memory:

- **Sheet "Summary"** — one row per employee (including employees with zero
  logs that month): `emp_code`, `name`, `days worked` (distinct working
  dates), `total hours` (sum of `working_duration`), `late count` (checkin
  time-of-day after `shift_settings.check_in_end`).
- **Sheet "Detail"** — one row per attendance log in the month: `emp_code`,
  `name`, `date`, `check-in`, `check-out`, `hours`.

Query: `Employee` LEFT JOIN `AttendanceLog` filtered by month. No
pagination — kiosk-scale data (hundreds of logs/month).

## Frontend

"Export Excel" button on the dashboard next to the existing month/year
selects, reusing the currently selected month. Click navigates to the report
URL; the browser downloads the file. No loading state.

## Testing

One backend test: seed 2 employees + logs, call the report service, assert
summary numbers (days, hours, late count) and that the workbook has both
sheets.

## Out of scope

Arbitrary date ranges, Excel styling, automatic email delivery.
