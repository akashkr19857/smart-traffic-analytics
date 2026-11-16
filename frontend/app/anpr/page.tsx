"use client";

import * as React from "react";
import { Loader2, Plus, Search, Trash2, Upload } from "lucide-react";
import { api, type TrafficEvent, type VehicleRow } from "@/lib/api";
import { EventRow } from "@/components/event-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

type Watch = { plate: string; reason: string; created_at: string };
type Recognized = { plates: { box: number[]; det_conf: number; text: string; ocr_conf: number }[]; ms: number };

export default function AnprPage() {
  const [reads, setReads] = React.useState<TrafficEvent[]>([]);
  const [q, setQ] = React.useState("");
  const [result, setResult] = React.useState<{ query: string; sightings: TrafficEvent[]; passages: VehicleRow[] } | null>(null);
  const [watch, setWatch] = React.useState<Watch[]>([]);
  const [plate, setPlate] = React.useState("");
  const [reason, setReason] = React.useState("");
  const [rec, setRec] = React.useState<Recognized | null>(null);
  const [img, setImg] = React.useState<string>("");
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState("");

  const loadWatch = () => api<Watch[]>("/api/watchlist").then(setWatch);
  React.useEffect(() => {
    api<TrafficEvent[]>("/api/events?type=plate,watchlist&limit=60").then(setReads);
    loadWatch();
  }, []);

  const search = async () => {
    if (q.trim().length < 2) return;
    setResult(await api(`/api/plates/search?q=${encodeURIComponent(q)}`));
  };
  const add = async () => {
    setError("");
    try {
      await api("/api/watchlist", { method: "POST", body: JSON.stringify({ plate, reason }) });
      setPlate("");
      setReason("");
      loadWatch();
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const upload = async (f: File) => {
    setBusy(true);
    setError("");
    setImg(URL.createObjectURL(f));
    const fd = new FormData();
    fd.append("file", f);
    try {
      setRec(await api<Recognized>("/api/anpr/recognize", { method: "POST", body: fd }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-4">
      <div><h1 className="text-xl font-semibold">ANPR &amp; watchlist</h1><p className="text-sm text-muted-foreground">Plates read from video are voted across several frames of the same tracked vehicle before they are logged.</p></div>
      {error && <p className="text-sm text-destructive">{error}</p>}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>Search a plate</CardTitle><CardDescription>Partial plates work, e.g. &quot;412&quot;</CardDescription></CardHeader>
          <CardContent className="space-y-3">
            <div className="flex gap-2">
              <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Plate number" data-testid="plate-search" onKeyDown={(e) => e.key === "Enter" && search()} />
              <Button onClick={search} data-testid="plate-search-go"><Search className="h-4 w-4" />Search</Button>
            </div>
            {result && (
              <div className="space-y-2" data-testid="plate-results">
                <p className="text-xs text-muted-foreground">{result.sightings.length} sightings · {result.passages.length} vehicle passages for &quot;{result.query}&quot;</p>
                {result.sightings.slice(0, 8).map((e) => <EventRow key={e.id} e={e} compact />)}
              </div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Watchlist</CardTitle><CardDescription>A critical alert fires when a listed plate is read on any camera</CardDescription></CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-wrap gap-2">
              <Input className="w-36" value={plate} onChange={(e) => setPlate(e.target.value)} placeholder="Plate" data-testid="watch-plate" />
              <Input className="min-w-40 flex-1" value={reason} onChange={(e) => setReason(e.target.value)} placeholder="Reason (e.g. stolen vehicle)" />
              <Button onClick={add} disabled={!plate.trim()} data-testid="watch-add"><Plus className="h-4 w-4" />Add</Button>
            </div>
            <div className="space-y-1" data-testid="watchlist">
              {watch.map((w) => (
                <div key={w.plate} className="flex items-center gap-2 rounded-md border p-2 text-sm">
                  <span className="rounded bg-foreground px-1.5 font-mono font-semibold text-background">{w.plate}</span>
                  <span className="min-w-0 flex-1 truncate text-muted-foreground">{w.reason}</span>
                  <Button size="icon" variant="ghost" className="h-7 w-7" aria-label={`Remove ${w.plate}`} onClick={() => api(`/api/watchlist/${w.plate}`, { method: "DELETE" }).then(loadWatch)}><Trash2 className="h-4 w-4" /></Button>
                </div>
              ))}
              {!watch.length && <p className="text-sm text-muted-foreground">No plates on the watchlist.</p>}
            </div>
          </CardContent>
        </Card>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>Recognise plates in an image</CardTitle><CardDescription>Upload a photo: plate detector + OCR, same models as the live pipeline</CardDescription></CardHeader>
          <CardContent className="space-y-3">
            <label className="inline-flex cursor-pointer items-center gap-2 rounded-md border px-3 py-2 text-sm hover:bg-accent">
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}Choose image
              <input type="file" accept="image/*" className="hidden" data-testid="anpr-upload" onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />
            </label>
            {img && (
              <div className="relative">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={img} alt="uploaded" className="w-full rounded border" />
              </div>
            )}
            {rec && (
              <div className="flex flex-wrap gap-2" data-testid="anpr-result">
                {rec.plates.map((p, i) => <Badge key={i} variant="default" className="font-mono text-sm">{p.text || "(unreadable)"} · det {(p.det_conf * 100).toFixed(0)}% · ocr {(p.ocr_conf * 100).toFixed(0)}%</Badge>)}
                {!rec.plates.length && <span className="text-sm text-muted-foreground">No plate found.</span>}
                <span className="text-xs text-muted-foreground">{rec.ms} ms</span>
              </div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Latest plate reads</CardTitle></CardHeader>
          <CardContent className="max-h-[520px] space-y-2 overflow-y-auto" data-testid="plate-reads">
            {reads.map((e) => <EventRow key={e.id} e={e} />)}
            {!reads.length && <p className="text-sm text-muted-foreground">No plates read yet.</p>}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
