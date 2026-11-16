"use client";

import * as React from "react";
import { Bar, BarChart, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type VehicleRow } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select } from "@/components/ui/input";

type A = {
  vehicles: number; by_class: Record<string, number>; per_minute: Record<string, number | string>[]; speed_histogram: { bucket: string; count: number }[];
  avg_speed_kmh: number | null; events_by_type: Record<string, number>; distinct_plates: number; plates_read: number;
};
const COLORS: Record<string, string> = { car: "hsl(152 60% 42%)", truck: "hsl(28 92% 52%)", bus: "hsl(350 75% 55%)", motorcycle: "hsl(205 85% 50%)" };
const tick = { fontSize: 11, fill: "hsl(var(--muted-foreground))" };

export default function AnalyticsPage() {
  const [cam, setCam] = React.useState("");
  const [a, setA] = React.useState<A | null>(null);
  const [rows, setRows] = React.useState<VehicleRow[]>([]);
  React.useEffect(() => {
    const load = () => {
      api<A>(`/api/analytics${cam ? `?camera_id=${cam}` : ""}`).then(setA);
      api<VehicleRow[]>(`/api/vehicles?limit=60${cam ? `&camera_id=${cam}` : ""}`).then(setRows);
    };
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [cam]);
  if (!a) return null;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div><h1 className="text-xl font-semibold">Traffic analytics</h1><p className="text-sm text-muted-foreground">Last 2 hours of vehicle passages (refreshes every 5 s).</p></div>
        <Select value={cam} onChange={(e) => setCam(e.target.value)} aria-label="Camera">
          <option value="">All cameras</option><option value="highway-d1">highway-d1</option><option value="highway-incidents">highway-incidents (simulated)</option><option value="expressway-city">expressway-city</option>
        </Select>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4" data-testid="analytics-kpis">
        {[["Vehicles", a.vehicles], ["Median speed", a.avg_speed_kmh ? `${a.avg_speed_kmh} km/h` : "–"], ["Plates read", a.plates_read],
          ["Violations", Object.entries(a.events_by_type).filter(([k]) => k !== "plate").reduce((s, [, v]) => s + v, 0)]].map(([l, v]) => (
          <div key={l as string} className="rounded-lg border bg-card p-3"><div className="text-xs text-muted-foreground">{l}</div><div className="text-2xl font-semibold tabular-nums">{v}</div></div>
        ))}
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card><CardHeader><CardTitle>Vehicles per minute by class</CardTitle></CardHeader>
          <CardContent className="h-64"><ResponsiveContainer><BarChart data={a.per_minute}><XAxis dataKey="minute" tick={tick} /><YAxis allowDecimals={false} tick={tick} /><Tooltip cursor={{ fill: "hsl(var(--muted))" }} /><Legend />
            {Object.keys(COLORS).map((c) => <Bar key={c} dataKey={c} stackId="a" fill={COLORS[c]} />)}</BarChart></ResponsiveContainer></CardContent></Card>
        <Card><CardHeader><CardTitle>Speed distribution</CardTitle><CardDescription>Median speed per vehicle, km/h (calibrated cameras only)</CardDescription></CardHeader>
          <CardContent className="h-64"><ResponsiveContainer><BarChart data={a.speed_histogram}><XAxis dataKey="bucket" tick={tick} /><YAxis allowDecimals={false} tick={tick} /><Tooltip cursor={{ fill: "hsl(var(--muted))" }} /><Bar dataKey="count" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></CardContent></Card>
      </div>
      <Card><CardHeader><CardTitle>Recent vehicle passages</CardTitle></CardHeader>
        <CardContent className="overflow-x-auto p-0">
          <table className="w-full text-sm" data-testid="passages"><thead className="bg-muted/60 text-left text-xs text-muted-foreground"><tr><th className="p-2 pl-4">Camera</th><th className="p-2">Track</th><th className="p-2">Class</th><th className="p-2">Direction</th><th className="p-2 text-right">Speed</th><th className="p-2">Plate</th></tr></thead>
            <tbody>{rows.map((v) => (
              <tr key={v.id} className="border-t"><td className="p-2 pl-4 text-xs">{v.camera_id}</td><td className="p-2 tabular-nums">#{v.track_id}</td><td className="p-2"><Badge variant="outline">{v.cls}</Badge></td><td className="p-2 text-xs">{v.direction || "–"}</td>
                <td className="p-2 text-right tabular-nums">{v.avg_speed_kmh ? `${v.avg_speed_kmh} km/h` : "–"}</td><td className="p-2 font-mono text-xs">{v.plate || ""}</td></tr>
            ))}</tbody></table>
        </CardContent></Card>
    </div>
  );
}
