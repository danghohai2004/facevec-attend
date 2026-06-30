import type { AttendanceRecord, Employee } from "@/lib/types";

export const demoEmployees: Employee[] = [
  { id: "e-1001", name: "Avery Tran", emp_code: "EMP-1001" },
  { id: "e-1002", name: "Minh Hoang", emp_code: "EMP-1002" },
  { id: "e-1003", name: "Jordan Lee", emp_code: "EMP-1003" },
  { id: "e-1004", name: "Priya Singh", emp_code: "EMP-1004" },
];

export const summaryMetrics = {
  totalEmployees: 128,
  todaysAttendance: 94,
  averageWorkingHours: 7.6,
};

const monthLabels = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

export const availableYears = [2024, 2025];

export function getMonthlyStats(year: number) {
  const base = year === 2025 ? 2200 : 2050;
  return monthLabels.map((month, index) => {
    const attendance = Math.round(base + index * 48 + (index % 3) * 30);
    const averageHours = Number((7.2 + (index % 4) * 0.2).toFixed(1));
    const workingHours = Math.round(attendance * averageHours);
    return {
      month,
      attendance,
      workingHours,
      averageHours,
    };
  });
}

export function getDailyAverages(year: number, month: string) {
  const monthIndex = monthLabels.indexOf(month);
  const base = year === 2025 ? 7.4 : 7.2;
  return Array.from({ length: 30 }, (_, day) => {
    const variation = ((day + monthIndex) % 5) * 0.15;
    return {
      day: `${day + 1}`.padStart(2, "0"),
      averageHours: Number((base + variation).toFixed(2)),
    };
  });
}

export const recentAttendance: AttendanceRecord[] = [
  {
    id: "att-1",
    emp_id: "e-1001",
    name: "Avery Tran",
    date: "2025-05-28",
    check_in: "08:12",
    check_out: "17:15",
    working_hours: 8.1,
    status: "checkout",
  },
  {
    id: "att-2",
    emp_id: "e-1002",
    name: "Minh Hoang",
    date: "2025-05-28",
    check_in: "08:05",
    status: "checkin",
  },
  {
    id: "att-3",
    emp_id: "e-1003",
    name: "Jordan Lee",
    date: "2025-05-28",
    check_in: "08:22",
    check_out: "17:05",
    working_hours: 7.7,
    status: "checkout",
  },
];
