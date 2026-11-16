"use client";

import * as React from "react";
import { api, eventsSocket, snapshotUrl, type TrafficEvent } from "@/lib/api";
import { EventBadge, EventRow } from "@/components/event-card";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select } from "@/components/ui/input";

const TYPES = [["", "All events"], ["wrong_way,illegal_parking,speeding,watchlist", "Violations only"], ["wrong_way", "Wrong way"],
  ["illegal_parking", "Illegal parking"], ["speeding", "Speeding"], ["plate", "Plate reads"], ["watchlist", "Watchlist hits"]];

export default function EventsPage() {
  const [type, setType] = React.useState("wrong_way,illegal_parking,speeding,watchlist");
  const [cam, setCam] = React.useState("");
  const [events, setEvents] = React.useState<TrafficEvent[]>([]);
  const [sel, setSel] = React.useState<TrafficEvent | null>(null);
  const load = React.useCallback(() => {
    const q = new URLSearchParams({ limit: "200" });
    if (type) q.set("type", type);
    if (cam) q.set("camera_id", cam);
    api<TrafficEvent[]>(`/api/events?${q}`).then((e) => {
      setEvents(e);
      setSel((s) => s ?? e[0] ?? null);
    });
  }, [type, cam]);
  React.useEffect(() => {
    load();
    return eventsSocket((ev) => ev.type === "event" && load());
  }, [load]);
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div><h1 className="text-xl font-semibold">Violations &amp; events</h1><p className="text-sm text-muted-foreground">Every event has an evidence snapshot with the vehicle boxed; plate reads include the plate crop.</p></div>
        <div className="flex gap-2">
          <Select value={type} onChange={(e) => setType(e.target.value)} aria-label="Event type" data-testid="event-type">{TYPES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</Select>
          <Select value={cam} onChange={(e) => setCam(e.target.value)} aria-label="Camera">
            <option value="">All cameras</option><option value="highway-d1">highway-d1</option><option value="highway-incidents">highway-incidents (simulated)</option><option value="expressway-city">expressway-city</option>
          </Select>
        </div>
      </div>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(360px,44%)]">
        <Card><CardContent className="space-y-2 pt-4" data-testid="event-list">
          {events.map((e) => <EventRow key={e.id} e={e} onOpen={setSel} />)}
          {!events.length && <p className="text-sm text-muted-foreground">No events of this type yet.</p>}
        </CardContent></Card>
        <Card className="h-fit xl:sticky xl:top-4">
          <CardHeader><CardTitle>Evidence</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            {sel ? (
              <>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={snapshotUrl(sel.snapshot)} alt="evidence snapshot" className="w-full rounded border" data-testid="evidence" />
                <div className="flex flex-wrap items-center gap-2"><EventBadge type={sel.type} /><span className="text-sm">{sel.message}</span></div>
                <dl className="grid grid-cols-2 gap-2 text-xs">
                  <div><dt className="text-muted-foreground">Camera</dt><dd>{sel.camera_id}</dd></div>
                  <div><dt className="text-muted-foreground">Video time</dt><dd>{sel.video_t.toFixed(1)} s</dd></div>
                  <div><dt className="text-muted-foreground">Vehicle</dt><dd>{sel.cls} · track #{sel.track_id}</dd></div>
                  <div><dt className="text-muted-foreground">Speed</dt><dd>{sel.speed_kmh ? `${sel.speed_kmh.toFixed(0)} km/h` : "–"}</dd></div>
                  <div><dt className="text-muted-foreground">Plate</dt><dd className="font-mono">{sel.plate || "not read"}{sel.plate ? ` (${(sel.plate_conf * 100).toFixed(0)}%)` : ""}</dd></div>
                  <div><dt className="text-muted-foreground">Logged</dt><dd>{new Date(sel.created_at).toLocaleTimeString()}</dd></div>
                </dl>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                {sel.plate_crop && <img src={snapshotUrl(sel.plate_crop)} alt="plate crop" className="h-12 rounded border" />}
              </>
            ) : <p className="text-sm text-muted-foreground">Select an event.</p>}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
