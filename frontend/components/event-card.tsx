"use client";

import { Badge } from "@/components/ui/badge";
import { EVENT_META, snapshotUrl, type TrafficEvent } from "@/lib/api";
import { cn } from "@/lib/utils";

export function EventBadge({ type }: { type: string }) {
  const m = EVENT_META[type] ?? { label: type, variant: "secondary" as const };
  return <Badge variant={m.variant}>{m.label}</Badge>;
}

/** Compact event row: snapshot thumbnail, plate crop, message, camera and video time. */
export function EventRow({ e, onOpen, compact }: { e: TrafficEvent; onOpen?: (e: TrafficEvent) => void; compact?: boolean }) {
  return (
    <button onClick={() => onOpen?.(e)} data-testid="event-row"
      className={cn("flex w-full items-center gap-3 rounded-md border p-2 text-left transition-colors hover:bg-accent", e.severity === "critical" && "border-destructive/50")}>
      {e.snapshot && (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={snapshotUrl(e.snapshot)} alt="" className={cn("shrink-0 rounded object-cover", compact ? "h-10 w-16" : "h-14 w-24")} loading="lazy" />
      )}
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-1.5"><EventBadge type={e.type} />
          {e.plate && <span className="rounded bg-foreground px-1.5 font-mono text-xs font-semibold text-background" data-testid="plate-chip">{e.plate}</span>}
        </div>
        <div className="truncate text-xs text-muted-foreground">{e.message}</div>
      </div>
      {e.plate_crop && !compact && (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={snapshotUrl(e.plate_crop)} alt="plate" className="h-8 shrink-0 rounded border" />
      )}
      <div className="shrink-0 text-right text-[11px] text-muted-foreground tabular-nums">{e.camera_id}<br />t={e.video_t.toFixed(1)}s</div>
    </button>
  );
}
