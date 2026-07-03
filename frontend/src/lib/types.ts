export type ShiftSettings = {
  checkInStart: string;
  checkInEnd: string;
  checkOutStart: string;
  checkOutEnd: string;
};

export type Employee = {
  id: string;
  name: string;
  emp_code: string;
  department?: string;
  created_at?: string;
  status?: "active" | "inactive";
};

export type EmployeeList = {
  items: Employee[];
  total: number;
  page: number;
  page_size: number;
};

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
