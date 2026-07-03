"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu } from "lucide-react";
import { ModeToggle } from "@/components/mode-toggle";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { navItems } from "@/components/layout/nav-items";

const pageTitles: Record<string, string> = {
  "/dashboard": "Dashboard & Analytics",
  "/employees": "Employee Management",
  "/shifts": "Shift Settings",
};

export function Topbar() {
  const pathname = usePathname();
  const title = pageTitles[pathname] ?? "Face Recognition Attendance";

  return (
    <header className="sticky top-0 z-40 flex h-16 items-center justify-between border-b-2 border-foreground bg-background px-6">
      <div className="flex items-center gap-3">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="icon" className="md:hidden">
              <Menu className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start">
            {navItems.map((item) => (
              <DropdownMenuItem key={item.title} asChild>
                <Link href={item.href}>{item.title}</Link>
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
        <div>
          <p className="text-sm font-black tracking-tight text-foreground uppercase">
            {title}
          </p>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <ModeToggle />
        <Avatar>
          <AvatarFallback className="bg-primary text-primary-foreground">
            AD
          </AvatarFallback>
        </Avatar>
      </div>
    </header>
  );
}
