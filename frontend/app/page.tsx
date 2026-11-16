"use client";

import * as React from "react";
import { Activity, Gauge, Radio, RotateCcw } from "lucide-react";
import { API_URL, api, eventsSocket, snapshotUrl, streamUrl, type Camera, type CamStats, type TrafficEvent } from "@/lib/api";
import { EventRow } from "@/components/event-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const CLASS_ORDER = ["car", "truck", "bus", "motorcycle"];

function CameraTile({ cam, stats, active, onSelect }: { cam: Camera; stats: CamStats; active: boolean; onSelect: () => void }) {
  // tiles poll a still frame once a second: browsers allow ~6 connections per host, so only the main view streams MJPEG
  const [tick, setTick] = React.useState(0);
  React.useEffect(() => {
    const t = setInterval(() => setTick((x) => x + 1), 1000);
    return () => clearInterval(t);
  }, []);
  return (
    <button onClick={onSelect} data-testid={`camera-${cam.id}`}
      className={cn("overflow-hidden rounded-lg border text-left transition-colors", active ? "border-primary ring-2 ring-primary/40" : "hover:border-primary/50")}>
      <div className="relative aspect-video bg-black">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={`${API_URL}/api/cameras/${cam.id}/snapshot.jpg?t=${tick}`} alt={cam.name} className="h-full w-full object-contain"
          onError={(e) => ((e.target as HTMLImageElement).style.visibility = "hidden")} onLoad={(e) => ((e.target as HTMLImageElement).style.visibility = "visible")} />
        <Badge variant={stats.status === "running" ? "success" : "secondary"} className="absolute left-2 top-2"><Radio className="h-3 w-3" />{stats.status}</Badge>
      </div>
      <div className="flex items-center justify-between gap-2 p-2 text-xs">
        <span className="truncate font-medium">{cam.name}</span>
        <span className="shrink-0 tabular-nums text-muted-foreground">{stats.passed ?? 0} veh · {stats.fps ?? 0} FPS</span>
      </div>
    </button>
  );
}

export default function LivePage() {
  const [cams, setCams] = React.useState<Camera[]>([]);
  const [stats, setStats] = React.useState<Record<string, CamStats>>({});
  const [sel, setSel] = React.useState<string | null>(null);
  const [events, setEvents] = React.useState<TrafficEvent[]>([]);
  const [open, setOpen] = React.useState<TrafficEvent | null>(null);
  const [error, setError] = React.useState("");

  React.useEffect(() => {
    api<Camera[]>("/api/cameras").then((c) => {
      setCams(c);
      setStats(Object.fromEntries(c.map((x) => [x.id, x.stats])));
      setSel((s) => s ?? c[0]?.id ?? null);
    }).catch((e) => setError(`API not reachable (${e.message}). Is the backend running on :8400?`));
    api<TrafficEvent[]>("/api/events?limit=30").then(setEvents).catch(() => {});
    return eventsSocket((ev) => {
      if (ev.type === "stats") setStats((s) => ({ ...s, [ev.camera_id as string]: ev.stats as CamStats }));
      else if (ev.type === "event") setEvents((es) => [ev.event as TrafficEvent, ...es].slice(0, 60));
    });
  }, []);

  const cam = cams.find((c) => c.id === sel);
  const st = (sel && stats[sel]) || ({} as CamStats);
  const camEvents = events.filter((e) => e.camera_id === sel);
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-xl font-semibold">Live cameras</h1>
          <p className="text-sm text-muted-foreground">Detection, ByteTrack tracking, speed, violations and ANPR running live on the GPU.</p>
        </div>
      </div>
      {error && <p className="text-sm text-destructive">{error}</p>}
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
        <div className="space-y-4">
          {cam && (
            <Card className="overflow-hidden">
              <div className="relative bg-black">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={streamUrl(cam.id)} alt={cam.name} className="max-h-[62vh] w-full object-contain" data-testid="main-stream" />
              </div>
              <CardContent className="space-y-3 pt-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="min-w-0">
                    <div className="font-semibold">{cam.name}</div>
                    <div className="text-xs text-muted-foreground">{cam.description}</div>
                  </div>
                  <Button size="sm" variant="outline" onClick={() => api(`/api/cameras/${cam.id}/restart`, { method: "POST" })}><RotateCcw className="h-4 w-4" />Restart</Button>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4" data-testid="live-stats">
                  <div className="rounded-md bg-muted/60 p-2"><div className="text-muted-foreground">Vehicles counted</div><div className="text-lg font-semibold tabular-nums">{st.passed ?? 0}</div></div>
                  <div className="rounded-md bg-muted/60 p-2"><div className="text-muted-foreground">On screen</div><div className="text-lg font-semibold tabular-nums">{st.active_tracks ?? 0}</div></div>
                  <div className="rounded-md bg-muted/60 p-2"><div className="flex items-center gap-1 text-muted-foreground"><Gauge className="h-3 w-3" />Median speed</div><div className="text-lg font-semibold tabular-nums">{st.avg_speed_kmh ? `${st.avg_speed_kmh} km/h` : "–"}</div></div>
                  <div className="rounded-md bg-muted/60 p-2"><div className="flex items-center gap-1 text-muted-foreground"><Activity className="h-3 w-3" />Pipeline</div><div className="text-lg font-semibold tabular-nums">{st.fps ?? 0} FPS</div></div>
                </div>
                <div className="flex flex-wrap gap-1.5 text-xs">
                  {CLASS_ORDER.map((c) => <Badge key={c} variant="outline">{c}: {st.counts?.[c] ?? 0}</Badge>)}
                  {cam.features.map((f) => <Badge key={f} variant="secondary">{f}</Badge>)}
                  {st.timings_ms && <span className="text-muted-foreground">detect {st.timings_ms.detect} ms · track {st.timings_ms.track} ms · rules+ANPR {st.timings_ms.rules_anpr} ms</span>}
                </div>
              </CardContent>
            </Card>
          )}
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {cams.map((c) => <CameraTile key={c.id} cam={c} stats={stats[c.id] ?? c.stats} active={c.id === sel} onSelect={() => setSel(c.id)} />)}
          </div>
        </div>
        <Card className="flex max-h-[calc(100vh-8rem)] flex-col">
          <CardHeader><CardTitle>Live events</CardTitle><CardDescription>{cam ? `${camEvents.length} on this camera` : ""} · newest first</CardDescription></CardHeader>
          <CardContent className="min-h-0 flex-1 space-y-2 overflow-y-auto" data-testid="live-events">
            {open && (
              <div className="space-y-2 rounded-md border p-2">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={snapshotUrl(open.snapshot)} alt="snapshot" className="w-full rounded" />
                <div className="text-sm">{open.message}</div>
                <Button size="sm" variant="ghost" onClick={() => setOpen(null)}>Close</Button>
              </div>
            )}
            {(camEvents.length ? camEvents : events).map((e) => <EventRow key={e.id} e={e} onOpen={setOpen} compact />)}
            {!events.length && <p className="text-sm text-muted-foreground">Events appear here as they happen.</p>}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
