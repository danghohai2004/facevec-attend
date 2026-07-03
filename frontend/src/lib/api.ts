import axios from "axios";
import type {
  DailyStats,
  Employee,
  EmployeeList,
  MonthlyStats,
  ShiftSettings,
  SummaryStats,
} from "@/lib/types";

const rawApiBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const normalizedApiBaseUrl = rawApiBaseUrl.replace(/\/+$/, "");
const apiBaseUrl = normalizedApiBaseUrl.endsWith("/api")
  ? normalizedApiBaseUrl
  : `${normalizedApiBaseUrl}/api`;

export const api = axios.create({
  baseURL: apiBaseUrl,
});

const writeApi = axios.create({
  baseURL: "/api/write",
});

function normalizeEmployee(raw: unknown): Employee {
  if (!raw || typeof raw !== "object") {
    throw new Error("Unexpected employee payload.");
  }

  const record = raw as Record<string, unknown>;
  const name = record.name;
  const empCode = record.emp_code ?? record.empCode;
  const id = record.id ?? record.emp_id ?? record.empId;

  if (
    typeof name !== "string" ||
    typeof empCode !== "string" ||
    (typeof id !== "string" && typeof id !== "number")
  ) {
    throw new Error("Unexpected employee payload.");
  }

  const status =
    record.status === "active" || record.status === "inactive"
      ? record.status
      : undefined;
  const department = typeof record.department === "string" ? record.department : undefined;
  const createdAt =
    typeof record.created_at === "string"
      ? record.created_at
      : typeof record.createdAt === "string"
        ? record.createdAt
        : undefined;

  return {
    id: String(id),
    name,
    emp_code: empCode,
    department,
    created_at: createdAt,
    status,
  };
}

function normalizeTimeValue(value: unknown): string {
  if (typeof value !== "string") {
    throw new Error("Unexpected shift settings payload.");
  }
  return value.length >= 5 ? value.slice(0, 5) : value;
}

function normalizeShiftSettings(raw: unknown): ShiftSettings {
  if (!raw || typeof raw !== "object") {
    throw new Error("Unexpected shift settings payload.");
  }

  const record = raw as Record<string, unknown>;
  const checkInStart = normalizeTimeValue(
    record.check_in_start ?? record.checkInStart,
  );
  const checkInEnd = normalizeTimeValue(
    record.check_in_end ?? record.checkInEnd,
  );
  const checkOutStart = normalizeTimeValue(
    record.check_out_start ?? record.checkOutStart,
  );
  const checkOutEnd = normalizeTimeValue(
    record.check_out_end ?? record.checkOutEnd,
  );

  return {
    checkInStart,
    checkInEnd,
    checkOutStart,
    checkOutEnd,
  };
}

function toApiShiftSettings(settings: ShiftSettings) {
  return {
    check_in_start: settings.checkInStart,
    check_in_end: settings.checkInEnd,
    check_out_start: settings.checkOutStart,
    check_out_end: settings.checkOutEnd,
  };
}

export async function listEmployees(params: {
  page: number;
  pageSize: number;
}): Promise<EmployeeList> {
  const response = await api.get("/employees", {
    params: { page: params.page, page_size: params.pageSize },
  });

  const payload = response.data;
  const rawItems = payload?.items ?? payload?.data ?? payload?.results ?? payload ?? [];

  if (!Array.isArray(rawItems)) {
    throw new Error("Unexpected employees response.");
  }

  const items = rawItems.map(normalizeEmployee);

  return {
    items,
    total: typeof payload?.total === "number" ? payload.total : items.length,
    page: params.page,
    page_size: params.pageSize,
  };
}

export async function getEmployeesByName(name: string): Promise<Employee[]> {
  try {
    const response = await api.get(`/employees/${encodeURIComponent(name)}`);
    const payload = response.data;
    const rawItems = payload?.items ?? payload?.data ?? payload?.results;

    if (Array.isArray(rawItems)) {
      return rawItems.map(normalizeEmployee);
    }

    const raw = payload?.employee ?? payload;
    if (raw && typeof raw === "object") {
      return [normalizeEmployee(raw)];
    }

    return [];
  } catch (error) {
    if (axios.isAxiosError(error) && error.response?.status === 404) {
      return [];
    }
    throw error;
  }
}

export async function createEmployee(payload: {
  name: string;
  empCode: string;
  files: File[];
}) {
  const formData = new FormData();
  formData.append("name", payload.name);
  formData.append("emp_code", payload.empCode);
  const representativeFile = payload.files[0];
  if (representativeFile) {
    formData.append("file", representativeFile);
  }

  const response = await writeApi.post("/employees", formData);
  const data = response.data;
  if (data?.employee) {
    return { ...data, employee: normalizeEmployee(data.employee) };
  }
  return data;
}

export async function deleteEmployee(employeeId: string) {
  const parsedId = Number(employeeId);
  const params = Number.isNaN(parsedId)
    ? { emp_code: employeeId }
    : { emp_id: parsedId };
  const response = await writeApi.delete("/employees", { params });
  return response.data;
}

export async function getShiftSettings(): Promise<ShiftSettings> {
  const response = await api.get("/shift-settings");
  return normalizeShiftSettings(response.data);
}

export async function updateShiftSettings(
  payload: ShiftSettings,
): Promise<ShiftSettings> {
  const response = await writeApi.put(
    "/shift-settings",
    toApiShiftSettings(payload),
  );
  return normalizeShiftSettings(response.data);
}

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

export function attendanceReportUrl(year: number, month: number): string {
  return `${apiBaseUrl}/attendance/report?year=${year}&month=${month}`;
}

export async function getDailyStats(
  year: number,
  month: number,
): Promise<DailyStats> {
  const response = await api.get("/attendance/daily", {
    params: { year, month },
  });
  const data = response.data;
  return {
    items: data.items.map((item: Record<string, number>) => ({
      day: item.day,
      averageHours: item.average_hours,
    })),
  };
}
