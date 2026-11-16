"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { AlertTriangle, BarChart3, Car, FlaskConical, Moon, ScanText, Sun, Video } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Live cameras", icon: Video },
  { href: "/events", label: "Violations & events", icon: AlertTriangle },
  { href: "/anpr", label: "ANPR & watchlist", icon: ScanText },
  { href: "/analytics", label: "Traffic analytics", icon: BarChart3 },
  { href: "/benchmarks", label: "Benchmarks", icon: FlaskConical },
];

function ThemeToggle() {
  const [dark, setDark] = React.useState(true);
  React.useEffect(() => {
    let stored: string | null = null;
    try {
      stored = localStorage.getItem("traffic_theme");
    } catch {
      /* ignore */
    }
    const isDark = stored ? stored === "dark" : true;
    document.documentElement.classList.toggle("dark", isDark);
    setDark(isDark);
  }, []);
  const toggle = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem("traffic_theme", next ? "dark" : "light");
    } catch {
      /* ignore */
    }
  };
  return <Button variant="ghost" size="icon" onClick={toggle} aria-label="Toggle theme">{dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}</Button>;
}

export function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  return (
    <div className="flex min-h-screen flex-col md:flex-row">
      <aside className="flex shrink-0 flex-row items-center gap-1 border-b bg-card px-3 py-2 md:w-56 md:flex-col md:items-stretch md:border-b-0 md:border-r md:py-4">
        <Link href="/" className="mb-0 mr-3 flex items-center gap-2 px-2 text-lg font-bold md:mb-6 md:mr-0">
          <Car className="h-6 w-6 text-primary" />
          <span className="leading-tight">Smart Traffic<br /><span className="text-xs font-medium text-muted-foreground">Analytics · ANPR</span></span>
        </Link>
        <nav className="flex flex-1 flex-row gap-1 overflow-x-auto md:flex-col">
          {NAV.map(({ href, label, icon: Icon }) => {
            const active = href === "/" ? path === "/" : path.startsWith(href);
            return (
              <Link key={href} href={href} className={cn("flex items-center gap-2 whitespace-nowrap rounded-md px-3 py-2 text-sm transition-colors",
                active ? "bg-primary/12 font-medium text-primary" : "text-muted-foreground hover:bg-accent hover:text-foreground")}>
                <Icon className="h-4 w-4" /><span className="hidden sm:inline">{label}</span>
              </Link>
            );
          })}
        </nav>
        <div className="hidden px-2 text-[11px] leading-relaxed text-muted-foreground md:block">
          YOLO11 · ByteTrack · RapidOCR
          <br />
          FastAPI · NVIDIA GPU (FP16)
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 items-center justify-end gap-2 border-b px-4"><ThemeToggle /></header>
        <main className="min-w-0 flex-1 p-4 md:p-6">{children}</main>
      </div>
    </div>
  );
}
