"use client";

import * as React from "react";
import { api } from "@/lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { pct } from "@/lib/utils";

type B = {
  plate_detector?: { epochs: number; train_minutes: number; test_split: { images: number; mAP50: number; mAP50_95: number; precision: number; recall: number } };
  anpr?: {
    images: number; us: number; eu: number;
    ocr_gt_crops: { raw: Record<string, number>; enhanced: Record<string, number>; ocr_ms_per_plate: number };
    simulated_video: { frames_per_plate: number; single_frame_accuracy: number; voted_accuracy: number; note: string };
    detection?: { recall_iou50: number; precision_iou50: number; detector_ms_per_image: number };
    end_to_end_exact_match?: number;
  };
  preprocess_ablation?: Record<string, { clean: number; degraded: number }>;
  rules?: {
    normal_footage: Record<string, unknown> & { speed_kmh: { vehicles: number; median: number; p10: number; p90: number } };
    simulated_incidents: { wrong_way: Record<string, number | null>; illegal_parking: Record<string, number | null>; note: string };
  };
  fps?: { rows: { config: string; ms_per_frame: number; fps: number }[]; hardware: string };
};

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return <div className="rounded-lg border bg-card p-3" data-testid="bench-metric"><div className="text-xs text-muted-foreground">{label}</div><div className="text-2xl font-semibold tabular-nums">{value}</div>{sub && <div className="text-[11px] text-muted-foreground">{sub}</div>}</div>;
}

export default function BenchmarksPage() {
  const [b, setB] = React.useState<B | null>(null);
  React.useEffect(() => {
    api<B>("/api/benchmarks").then(setB);
  }, []);
  if (!b) return null;
  const d = b.plate_detector, a = b.anpr, r = b.rules;
  return (
    <div className="space-y-4">
      <div><h1 className="text-xl font-semibold">Benchmarks</h1><p className="text-sm text-muted-foreground">Measured on an NVIDIA RTX 4060 Laptop GPU. Scripts in backend/eval/ and models/.</p></div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {d && <Stat label="Plate detector mAP@50" value={pct(d.test_split.mAP50, 1)} sub={`held-out test split, ${d.test_split.images} images`} />}
        {a?.detection && <Stat label="Plate detection recall" value={pct(a.detection.recall_iou50, 1)} sub={`OpenALPR benchmark, ${a.images} unseen images`} />}
        {a && <Stat label="OCR on plate crops" value={pct(a.ocr_gt_crops.enhanced.all, 1)} sub={`exact match, raw crop ${pct(a.ocr_gt_crops.raw.all, 1)}`} />}
        {a && <Stat label="Voting over 5 frames" value={pct(a.simulated_video.voted_accuracy, 1)} sub={`vs ${pct(a.simulated_video.single_frame_accuracy, 1)} single frame (simulated)`} />}
        {a?.end_to_end_exact_match !== undefined && <Stat label="End-to-end ANPR" value={pct(a.end_to_end_exact_match, 1)} sub="detector + OCR, exact plate" />}
        {r && <Stat label="Wrong-way detection" value={pct(r.simulated_incidents.wrong_way.detection_rate as number, 0)} sub={`${r.simulated_incidents.wrong_way.flagged}/${r.simulated_incidents.wrong_way.moving_vehicles_in_reversed_segment} vehicles, simulated`} />}
        {r && <Stat label="False alarms (real footage)" value={String((r.normal_footage.wrong_way_false_alarms as number) + (r.normal_footage.illegal_parking_false_alarms as number))} sub="wrong-way + parking" />}
        {r && <Stat label="Pipeline speed (4K)" value={`${r.normal_footage.end_to_end_fps} FPS`} sub={`${r.normal_footage.ms_per_frame_p50} ms per frame incl. ANPR`} />}
      </div>
      {b.preprocess_ablation && (
        <Card><CardHeader><CardTitle>OCR preprocessing ablation</CardTitle><CardDescription>Exact-match accuracy on 330 labelled plates. Blanket preprocessing hurt, so the pipeline reads raw, deskewed and CLAHE+deskewed candidates and keeps the most confident.</CardDescription></CardHeader>
          <CardContent className="overflow-x-auto"><table className="w-full text-sm"><thead className="text-left text-xs text-muted-foreground"><tr><th className="py-1 pr-3">Variant</th><th className="pr-3">Clean crops</th><th>Degraded crops</th></tr></thead>
            <tbody>{Object.entries(b.preprocess_ablation).map(([k, v]) => <tr key={k} className="border-t"><td className="py-1.5 pr-3">{k}</td><td className="pr-3 tabular-nums">{pct(v.clean, 1)}</td><td className="tabular-nums">{pct(v.degraded, 1)}</td></tr>)}</tbody></table></CardContent></Card>
      )}
      {r && (
        <Card><CardHeader><CardTitle>Traffic rules on video</CardTitle><CardDescription>{r.simulated_incidents.note}</CardDescription></CardHeader>
          <CardContent className="grid gap-4 text-sm md:grid-cols-2">
            <div><div className="mb-1 font-medium">Real highway footage</div><pre className="overflow-x-auto rounded bg-muted/60 p-2 text-xs">{JSON.stringify(r.normal_footage, null, 1)}</pre></div>
            <div><div className="mb-1 font-medium">Simulated incidents</div><pre className="overflow-x-auto rounded bg-muted/60 p-2 text-xs">{JSON.stringify({ wrong_way: r.simulated_incidents.wrong_way, illegal_parking: r.simulated_incidents.illegal_parking }, null, 1)}</pre></div>
          </CardContent></Card>
      )}
      {b.fps && (
        <Card><CardHeader><CardTitle>Inference speed</CardTitle><CardDescription>{b.fps.hardware}</CardDescription></CardHeader>
          <CardContent><table className="w-full text-sm"><thead className="text-left text-xs text-muted-foreground"><tr><th className="py-1 pr-3">Configuration</th><th className="pr-3">ms / frame</th><th>FPS</th></tr></thead>
            <tbody>{b.fps.rows.map((x) => <tr key={x.config} className="border-t"><td className="py-1.5 pr-3">{x.config}</td><td className="pr-3 tabular-nums">{x.ms_per_frame}</td><td className="tabular-nums">{x.fps}</td></tr>)}</tbody></table></CardContent></Card>
      )}
    </div>
  );
}
