"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { navItems } from "@/components/layout/nav-items";

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden md:flex md:w-64 md:flex-col md:border-r-2 md:border-foreground md:bg-sidebar">
      <div className="flex h-16 items-center px-6">
        <div className="flex items-center gap-2.5">
          <div className="flex h-10 w-10 items-center justify-center rounded-[3px] border-2 border-foreground bg-primary text-sm font-black text-primary-foreground shadow-brutal-sm">
            FA
          </div>
          <p className="text-sm leading-none font-black tracking-tight uppercase">
            FaceVec Attend
          </p>
        </div>
      </div>
      <nav className="flex flex-1 flex-col gap-2 px-3 py-4">
        {navItems.map((item) => {
          const active = pathname === item.href;
          const Icon = item.icon;
          return (
            <Link
              key={item.title}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-[3px] border-2 px-3 py-2 text-sm font-bold tracking-wide uppercase transition-all",
                active
                  ? "border-foreground bg-sidebar-accent text-sidebar-accent-foreground shadow-brutal-sm"
                  : "border-transparent text-muted-foreground hover:border-foreground hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground",
              )}
            >
              <Icon className="h-4 w-4" />
              <span>{item.title}</span>
            </Link>
          );
        })}
      </nav>
      <div className="px-6 pb-6">
        <div className="rounded-[3px] border-2 border-foreground bg-secondary p-4 text-sm text-secondary-foreground shadow-brutal-sm">
          <p className="font-black tracking-tight uppercase">
            Face Recognition System
          </p>
          <p className="mt-1 text-xs font-bold">
            Secure biometric attendance with realtime monitoring.
          </p>
        </div>
      </div>
    </aside>
  );
}
