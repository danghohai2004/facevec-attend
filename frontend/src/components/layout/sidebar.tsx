"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { navItems } from "@/components/layout/nav-items";

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden md:flex md:w-64 md:flex-col md:border-r md:bg-card/60">
      <div className="flex h-16 items-center justify-between px-6">
        <div className="flex items-center gap-2">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary text-primary-foreground">
            FA
          </div>
          <div>
            <p className="text-sm font-semibold leading-none">FaceVec Attend</p>
            <p className="text-xs text-muted-foreground">Enterprise Dashboard</p>
          </div>
        </div>
        <Badge variant="secondary">Live</Badge>
      </div>
      <nav className="flex flex-1 flex-col gap-1 px-3 py-4">
        {navItems.map((item) => {
          const active = pathname === item.href;
          const Icon = item.icon;
          if (item.disabled) {
            return (
              <div
                key={item.title}
                className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm text-muted-foreground"
              >
                <Icon className="h-4 w-4" />
                <span>{item.title}</span>
                <Badge variant="outline" className="ml-auto text-xs">
                  Soon
                </Badge>
              </div>
            );
          }
          return (
            <Link
              key={item.title}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
                active
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:bg-accent/60 hover:text-accent-foreground",
              )}
            >
              <Icon className="h-4 w-4" />
              <span>{item.title}</span>
            </Link>
          );
        })}
      </nav>
      <div className="px-6 pb-6">
        <div className="rounded-xl border bg-background p-4 text-sm text-muted-foreground">
          <p className="font-medium text-foreground">
            Face Recognition System
          </p>
          <p className="mt-1 text-xs">
            Secure biometric attendance with realtime monitoring.
          </p>
        </div>
      </div>
    </aside>
  );
}
