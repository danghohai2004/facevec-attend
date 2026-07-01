import {
  Clock3,
  LayoutDashboard,
  Settings,
  Users,
  Video,
} from "lucide-react";

export const navItems = [
  {
    title: "Dashboard",
    href: "/dashboard",
    icon: LayoutDashboard,
  },
  {
    title: "Attendance",
    href: "/attendance",
    icon: Video,
  },
  {
    title: "Employees",
    href: "/employees",
    icon: Users,
  },
  {
    title: "Shift Settings",
    href: "/shifts",
    icon: Clock3,
  },
  {
    title: "System Settings",
    href: "/settings",
    icon: Settings,
    disabled: true,
  },
];
