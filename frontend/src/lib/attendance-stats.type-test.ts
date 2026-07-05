import type {
  DailyStats,
  MonthlyStats,
  SummaryStats,
} from "@/lib/types";
import {
  getDailyStats,
  getMonthlyStats,
  getSummaryStats,
} from "@/lib/api";

const summary = {
  totalEmployees: 128,
  todaysAttendance: 94,
  averageWorkingHours: 7.6,
  onTimeRate: 91.5,
  deltas: {
    todaysAttendance: 2.1,
    averageWorkingHours: null,
    onTimeRate: 3.4,
  },
} satisfies SummaryStats;

const monthly = {
  availableYears: [2025, 2026],
  items: [
    {
      month: 7,
      attendance: 94,
      workingHours: 714.5,
      averageHours: 7.6,
    },
  ],
} satisfies MonthlyStats;

const daily = {
  items: [{ day: 3, averageHours: 7.58 }],
} satisfies DailyStats;

void summary;
void monthly;
void daily;

const summaryRequest: () => Promise<SummaryStats> = getSummaryStats;
const monthlyRequest: (year: number) => Promise<MonthlyStats> =
  getMonthlyStats;
const dailyRequest: (
  year: number,
  month: number,
) => Promise<DailyStats> = getDailyStats;

void summaryRequest;
void monthlyRequest;
void dailyRequest;
