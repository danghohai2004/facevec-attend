"use client";

import * as React from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ArrowDownRight,
  ArrowUpRight,
  Clock,
  TrendingUp,
  UserCheck,
  Users,
  type LucideIcon,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { getDailyStats, getMonthlyStats, getSummaryStats } from "@/lib/api";
import { formatHours } from "@/lib/format";
import { cn } from "@/lib/utils";

const monthOptions = [
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

// Hard, non-blurred tooltip to match the poster/comic card language.
const tooltipStyle = {
  borderRadius: "3px",
  border: "2px solid var(--foreground)",
  background: "var(--popover)",
  color: "var(--popover-foreground)",
  fontSize: "0.75rem",
  fontWeight: 700,
  boxShadow: "4px 4px 0 0 var(--foreground)",
} as const;

type PosterAccent = "yellow" | "cyan" | "pink" | "lime";

const posterBg: Record<PosterAccent, string> = {
  yellow: "bg-poster-yellow",
  cyan: "bg-poster-cyan",
  pink: "bg-poster-pink",
  lime: "bg-poster-lime",
};

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
  const isError = summaryQuery.isError || monthlyQuery.isError || dailyQuery.isError;

  return (
    <div className="space-y-6">
      {isError && (
        <div className="rounded-[3px] border-2 border-destructive bg-destructive/10 px-4 py-3 text-sm font-bold text-destructive">
          Unable to load dashboard data.
        </div>
      )}
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

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-black tracking-tight uppercase">
            Attendance Analytics
          </h2>
        </div>
        <div className="flex flex-wrap gap-3">
          <Select value={selectedYear} onValueChange={setSelectedYear}>
            <SelectTrigger className="w-[140px]">
              <SelectValue placeholder="Year" />
            </SelectTrigger>
            <SelectContent>
              {availableYears.map((year) => (
                <SelectItem key={year} value={`${year}`}>
                  {year}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={selectedMonth} onValueChange={setSelectedMonth}>
            <SelectTrigger className="w-[140px]">
              <SelectValue placeholder="Month" />
            </SelectTrigger>
            <SelectContent>
              {monthOptions.map((month) => (
                <SelectItem key={month} value={month}>
                  {month}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Monthly Attendance Trend</CardTitle>
          </CardHeader>
          <CardContent className="h-[320px]">
            <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 800, height: 320 }}>
              <AreaChart data={monthlyStats} margin={{ left: -12, right: 8, top: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--muted-foreground)" strokeOpacity={0.25} vertical={false} />
                <XAxis dataKey="month" tickLine={false} axisLine={false} fontSize={12} fontWeight={700} stroke="var(--foreground)" />
                <YAxis tickLine={false} axisLine={false} fontSize={12} fontWeight={700} stroke="var(--foreground)" width={48} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ stroke: "var(--foreground)", strokeWidth: 2, strokeDasharray: "4 4" }} />
                <Area
                  type="monotone"
                  dataKey="attendance"
                  stroke="var(--foreground)"
                  strokeWidth={3}
                  fill="var(--chart-1)"
                  fillOpacity={1}
                  activeDot={{ r: 5, stroke: "var(--foreground)", strokeWidth: 2 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Total Working Hours</CardTitle>
          </CardHeader>
          <CardContent className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 400, height: 280 }}>
              <BarChart data={monthlyStats} margin={{ left: -12, right: 8, top: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--muted-foreground)" strokeOpacity={0.25} vertical={false} />
                <XAxis dataKey="month" tickLine={false} axisLine={false} fontSize={12} fontWeight={700} stroke="var(--foreground)" />
                <YAxis tickLine={false} axisLine={false} fontSize={12} fontWeight={700} stroke="var(--foreground)" width={48} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--foreground)", fillOpacity: 0.08 }} />
                <Bar
                  dataKey="workingHours"
                  fill="var(--chart-2)"
                  stroke="var(--foreground)"
                  strokeWidth={2}
                  radius={[2, 2, 0, 0]}
                  maxBarSize={36}
                />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Average Hours per Day</CardTitle>
          </CardHeader>
          <CardContent className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 400, height: 280 }}>
              <AreaChart data={dailyAverages} margin={{ left: -12, right: 8, top: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--muted-foreground)" strokeOpacity={0.25} vertical={false} />
                <XAxis dataKey="day" tickLine={false} axisLine={false} fontSize={12} fontWeight={700} stroke="var(--foreground)" interval={4} />
                <YAxis tickLine={false} axisLine={false} fontSize={12} fontWeight={700} stroke="var(--foreground)" width={48} domain={[0, "dataMax + 1"]} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ stroke: "var(--foreground)", strokeWidth: 2, strokeDasharray: "4 4" }} />
                <Area
                  type="monotone"
                  dataKey="averageHours"
                  stroke="var(--foreground)"
                  strokeWidth={3}
                  fill="var(--chart-3)"
                  fillOpacity={1}
                  activeDot={{ r: 5, stroke: "var(--foreground)", strokeWidth: 2 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
