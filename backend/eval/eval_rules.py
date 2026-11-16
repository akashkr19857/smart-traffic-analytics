"""Evaluate the traffic rules end to end on video (local GPU):

* normal footage (vehicles.mp4, real): wrong-way and illegal-parking FALSE alarms (should be 0), speed distribution,
  vehicles counted, processing speed
* simulated incidents (data/scenarios/incidents.mp4): share of moving vehicles in the time-reversed segment flagged as
  wrong-way, and whether / when the pasted stationary car is flagged as illegally parked

    python backend/eval/eval_rules.py   → backend/eval/results/rules.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import load_cameras  # noqa: E402
from app.vision.pipeline import CameraPipeline, Models  # noqa: E402

from contextlib import nullcontext as gpu_lock  # single-process: no cross-process GPU lock needed


def run(cfg, models, path: Path, max_s: float | None = None) -> dict:  # noqa: ANN001
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    p = CameraPipeline(cfg, models, None, fps)
    events, frame_ms, i = [], [], 0
    t_start = time.perf_counter()
    while True:
        ok, f = cap.read()
        if not ok or (max_s and i / fps > max_s):
            break
        r = p.process(f, i / fps, annotate=False)
        frame_ms.append(r.timings_ms["total"])
        events += [{k: v for k, v in e.items() if k not in ("_crop", "box")} for e in r.events]
        i += 1
    wall = time.perf_counter() - t_start
    tracks = p.finished + list(p.tracks.values())
    return {"frames": i, "seconds": round(i / fps, 1), "events": events, "tracks": tracks, "pipeline": p,
            "ms_per_frame_p50": round(float(np.median(frame_ms[10:])), 1), "end_to_end_fps": round(i / wall, 1)}


def main() -> None:
    cams = {c.id: c for c in load_cameras()}
    cfg = cams["highway-d1"]
    truth = json.loads((ROOT / "data" / "scenarios" / "incidents.json").read_text(encoding="utf-8"))
    with gpu_lock("traffic-rules-eval"):
        models = Models.load(with_anpr=True)
        normal = run(cfg, models, ROOT / cfg.source)
        inc = run(cfg, models, ROOT / "data" / "scenarios" / "incidents.json".replace(".json", ".mp4"))
    speeds = [float(np.median(t.speeds)) for t in normal["tracks"] if len(t.speeds) >= 5]
    ev_n = normal["events"]
    res = {"normal_footage": {
        "video": cfg.source, "seconds": normal["seconds"], "vehicles_tracked": sum(len(t.image_pts) >= 10 for t in normal["tracks"]),
        "vehicles_counted": normal["pipeline"].stats.passed, "counts_by_class": dict(normal["pipeline"].stats.counts),
        "wrong_way_false_alarms": sum(e["type"] == "wrong_way" for e in ev_n),
        "illegal_parking_false_alarms": sum(e["type"] == "illegal_parking" for e in ev_n),
        "speeding_events": sum(e["type"] == "speeding" for e in ev_n),
        "plate_events": sum(e["type"] == "plate" for e in ev_n),
        "plates_read": sorted({e["plate"] for e in ev_n if e["type"] == "plate"}),
        "speed_kmh": {"vehicles": len(speeds), "median": round(float(np.median(speeds)), 1) if speeds else None,
                      "p10": round(float(np.percentile(speeds, 10)), 1) if speeds else None,
                      "p90": round(float(np.percentile(speeds, 90)), 1) if speeds else None},
        "ms_per_frame_p50": normal["ms_per_frame_p50"], "end_to_end_fps": normal["end_to_end_fps"]}}
    seg_end = truth["wrong_way_segment_s"][1]
    rev_tracks = [t for t in inc["tracks"] if t.first_ts < seg_end - 1 and len(t.image_pts) >= 15 and t.zone
                  and np.hypot(t.image_pts[-1][1] - t.image_pts[0][1], t.image_pts[-1][2] - t.image_pts[0][2]) > 60]
    flagged = [t for t in rev_tracks if t.wrong_way_reported]
    park = [e for e in inc["events"] if e["type"] == "illegal_parking"]
    pv = truth["parked_vehicle"]
    res["simulated_incidents"] = {
        "note": truth["note"],
        "wrong_way": {"moving_vehicles_in_reversed_segment": len(rev_tracks), "flagged": len(flagged),
                      "detection_rate": round(len(flagged) / len(rev_tracks), 3) if rev_tracks else None,
                      "wrong_way_events_after_segment": sum(e["type"] == "wrong_way" and e["video_t"] > seg_end + 2 for e in inc["events"])},
        "illegal_parking": {"parked_from_s": pv["from_s"], "events": len(park),
                            "first_alert_s": round(park[0]["video_t"], 1) if park else None,
                            "alert_delay_s": round(park[0]["video_t"] - pv["from_s"], 1) if park else None,
                            "configured_threshold_s": cfg.parking_seconds}}
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "rules.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
