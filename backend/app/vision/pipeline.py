"""Per-camera processing: detect → ByteTrack → speed → rules (count, speeding, wrong-way, illegal parking) → ANPR → events.

`CameraPipeline.process(frame, t)` is pure frame-in / results-out (no threads, no I/O except optional event callback), so
it is driven the same way by the live runner, the benchmark script and the tests.
"""
from __future__ import annotations

import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
import supervision as sv

from ..anpr.ocr import PlateOCR, PlateRead, vote
from ..config import CameraConfig, get_settings
from .tracking import ByteTrack
from .geometry import TrackState, ViewTransformer, displacement_px, heading_sign, point_in_polygon, side_of_line, speed_kmh

log = logging.getLogger("traffic.pipeline")
COCO_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CLASS_COLORS = {"car": (60, 180, 75), "truck": (245, 130, 48), "bus": (230, 25, 75), "motorcycle": (0, 130, 200)}


@dataclass
class FrameResult:
    t: float
    detections: int
    tracks: list[dict]
    events: list[dict]
    timings_ms: dict[str, float]
    annotated: np.ndarray | None = None


@dataclass
class Models:
    """Loaded once and shared by every camera (one GPU)."""
    vehicle: object
    plate: object | None
    ocr: PlateOCR | None

    @classmethod
    def load(cls, with_anpr: bool = True) -> "Models":
        from ultralytics import YOLO

        s = get_settings()
        veh = YOLO(s.vehicle_model)
        plate = YOLO(str(s.plate_model)) if with_anpr and s.plate_model.exists() else None
        return cls(veh, plate, PlateOCR() if plate is not None else None)


@dataclass
class Stats:
    counts: Counter = field(default_factory=Counter)  # by class, vehicles that crossed the count line
    passed: int = 0
    speeds: deque = field(default_factory=lambda: deque(maxlen=500))
    events: Counter = field(default_factory=Counter)
    fps: float = 0.0
    frames: int = 0


class CameraPipeline:
    def __init__(self, cfg: CameraConfig, models: Models, on_event: Callable[[dict, np.ndarray, np.ndarray | None], None] | None = None,
                 fps_hint: float = 25.0) -> None:
        self.cfg, self.m, self.on_event = cfg, models, on_event
        self.tracker = ByteTrack(frame_rate=int(round(fps_hint)), lost_track_buffer=int(fps_hint * 1.5))
        self.view = ViewTransformer(cfg.speed_source, cfg.speed_target_m) if cfg.speed_source and cfg.speed_target_m else None
        self.tracks: dict[int, TrackState] = {}
        self.finished: list[TrackState] = []  # tracks that left the view (persisted by the runner)
        self.stats = Stats()
        self._last = time.perf_counter()
        self.s = get_settings()
        self._q = 16 if self.s.half and self.s.device != "cpu" else 32  # FP16 inference on GPU

    # ------------------------------------------------------------------ detection
    def _detect(self, frame: np.ndarray) -> sv.Detections:
        h, w = frame.shape[:2]
        scale = self.cfg.infer_width / w
        small = cv2.resize(frame, (self.cfg.infer_width, int(h * scale))) if scale < 1 else frame
        r = self.m.vehicle.predict(small, conf=self.s.detect_conf, classes=self.cfg.vehicle_classes, imgsz=self.cfg.infer_width,
                                   device=self.s.device, quantize=self._q, verbose=False)[0]
        b = r.boxes
        det = sv.Detections(xyxy=b.xyxy.cpu().numpy().astype(np.float32).reshape(-1, 4),
                            confidence=b.conf.cpu().numpy().astype(np.float32), class_id=b.cls.cpu().numpy().astype(int))
        if scale < 1 and len(det):
            det.xyxy = det.xyxy / scale
        return det

    # ------------------------------------------------------------------ ANPR on one vehicle
    def _read_plate(self, frame: np.ndarray, box: np.ndarray) -> tuple[PlateRead | None, np.ndarray | None]:
        x1, y1, x2, y2 = box.astype(int)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        car = frame[y1:y2, x1:x2]
        if car.size == 0:
            return None, None
        r = self.m.plate.predict(car, conf=self.s.plate_conf, imgsz=640, device=self.s.device,
                                 quantize=self._q, verbose=False)[0]
        if not len(r.boxes):
            return None, None
        i = int(r.boxes.conf.argmax())
        px1, py1, px2, py2 = r.boxes.xyxy[i].cpu().numpy().astype(int)
        pad = int(0.06 * (px2 - px1))
        crop = car[max(0, py1 - pad):py2 + pad, max(0, px1 - pad):px2 + pad]
        read = self.m.ocr.read(crop)
        return (read if len(read.text) >= 4 else None), crop

    # ------------------------------------------------------------------ main step
    def process(self, frame: np.ndarray, t: float, annotate: bool = True) -> FrameResult:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        det = self._detect(frame)
        timings["detect"] = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        det = self.tracker.update_with_detections(det)
        timings["track"] = (time.perf_counter() - t1) * 1000
        events: list[dict] = []
        seen: set[int] = set()
        anchors = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(det) else np.zeros((0, 2))
        world = self.view.transform(anchors) if self.view is not None and len(det) else None
        t2 = time.perf_counter()
        anpr_ms = 0.0
        for i in range(len(det)):
            tid = int(det.tracker_id[i])
            seen.add(tid)
            cls = COCO_NAMES.get(int(det.class_id[i]), "vehicle")
            st = self.tracks.get(tid)
            if st is None:
                st = self.tracks[tid] = TrackState(tid, cls, t, t)
            st.last_ts, st.cls = t, cls
            ax, ay = map(float, anchors[i])
            st.image_pts.append((t, ax, ay))
            box = det.xyxy[i]
            st.box_h = float(box[3] - box[1])
            # --- speed
            if world is not None:
                st.world_pts.append((t, float(world[i][0]), float(world[i][1])))
                sp = speed_kmh(st.world_pts)
                if sp is not None and 3 < sp < 250:
                    st.speeds.append(sp)
                    lim = self.cfg.speed_limit_kmh
                    recent = st.speeds[-5:]
                    if lim and len(recent) >= 5 and min(recent) > lim and not st.speeding_reported:
                        st.speeding_reported = True
                        events.append(self._event("speeding", st, box, f"{cls} at {np.median(recent):.0f} km/h (limit {lim:.0f})",
                                                  severity="warning", speed=float(np.median(recent))))
            # --- counting line
            if self.cfg.count_line and not st.counted and len(st.image_pts) >= 2:
                a, b = self.cfg.count_line
                prev = st.image_pts[-2]
                if side_of_line((prev[1], prev[2]), a, b) * side_of_line((ax, ay), a, b) < 0:
                    st.counted = True
                    self.stats.counts[cls] += 1
                    self.stats.passed += 1
            # --- wrong way
            if self.cfg.vanishing_point:
                for z in self.cfg.direction_zones:
                    if point_in_polygon((ax, ay), z.polygon):
                        st.zone = z.name
                        sign = heading_sign(st.image_pts, self.cfg.vanishing_point)
                        legal = 1 if z.legal == "away" else -1
                        st.wrong_way_frames = st.wrong_way_frames + 1 if sign == -legal else max(0, st.wrong_way_frames - 1)
                        if st.wrong_way_frames >= 8 and not st.wrong_way_reported:
                            st.wrong_way_reported = True
                            events.append(self._event("wrong_way", st, box, f"{cls} driving against traffic in {z.name}", "critical"))
                        break
            # --- illegal parking (stationary inside a no-parking zone)
            in_np = any(point_in_polygon((ax, ay), zone) for zone in self.cfg.no_parking_zones)
            if in_np:
                if st.stationary_since is None:
                    st.stationary_since = t
                still = displacement_px(st.image_pts, st.stationary_since) < max(12.0, 0.08 * st.box_h)
                if not still:
                    st.stationary_since = t
                elif t - st.stationary_since >= self.cfg.parking_seconds and not st.parking_reported:
                    st.parking_reported = True
                    events.append(self._event("illegal_parking", st, box,
                                              f"{cls} stopped {t - st.stationary_since:.0f} s in a no-parking zone", "warning"))
            else:
                st.stationary_since = None
            # --- ANPR: vehicles near the camera, a few reads per track, then vote
            if self.cfg.anpr and self.m.plate is not None and st.box_h >= self.cfg.anpr_min_box_h and len(st.plate_reads) < 8 \
                    and len(st.image_pts) % 2 == 0:
                ta = time.perf_counter()
                read, crop = self._read_plate(frame, box)
                anpr_ms += (time.perf_counter() - ta) * 1000
                if read is not None:
                    st.plate_reads.append(read)
                    v = vote(st.plate_reads)
                    if v is not None:
                        changed = v.text != st.plate
                        st.plate, st.plate_conf = v.text, v.conf
                        if changed and len(st.plate_reads) >= 3:
                            events.append(self._event("plate", st, box, f"{cls} plate {v.text}", "info", crop=crop))
        timings["rules_anpr"] = (time.perf_counter() - t2) * 1000
        timings["anpr"] = anpr_ms
        # tracks that disappeared for > 2 s are finished
        for tid in [k for k, v in self.tracks.items() if k not in seen and t - v.last_ts > 2.0]:
            self.finished.append(self.tracks.pop(tid))
        now = time.perf_counter()
        self.stats.frames += 1
        self.stats.fps = 0.9 * self.stats.fps + 0.1 * (1.0 / max(now - self._last, 1e-6)) if self.stats.frames > 1 else 0.0
        self._last = now
        for st_ in self.tracks.values():
            if st_.speeds:
                self.stats.speeds.append(st_.speeds[-1])
        tracks = [{"track_id": int(det.tracker_id[i]), "cls": COCO_NAMES.get(int(det.class_id[i]), "vehicle"),
                   "box": [float(v) for v in det.xyxy[i]], "speed": round(self.tracks[int(det.tracker_id[i])].speeds[-1], 1)
                   if self.tracks.get(int(det.tracker_id[i])) and self.tracks[int(det.tracker_id[i])].speeds else None,
                   "plate": self.tracks[int(det.tracker_id[i])].plate if self.tracks.get(int(det.tracker_id[i])) else ""}
                  for i in range(len(det))]
        for e in events:
            self.stats.events[e["type"]] += 1
            if self.on_event:
                self.on_event(e, frame, e.pop("_crop", None))
        annotated = self.annotate(frame, tracks) if annotate else None
        timings["total"] = (time.perf_counter() - t0) * 1000
        return FrameResult(t, len(det), tracks, events, timings, annotated)

    def _event(self, etype: str, st: TrackState, box: np.ndarray, msg: str, severity: str, speed: float | None = None,
               crop: np.ndarray | None = None) -> dict:
        return {"type": etype, "severity": severity, "track_id": st.track_id, "cls": st.cls, "plate": st.plate,
                "plate_conf": st.plate_conf, "speed_kmh": speed if speed is not None else (st.speeds[-1] if st.speeds else None),
                "message": msg, "video_t": st.last_ts, "box": [float(v) for v in box], "zone": st.zone, "_crop": crop}

    # ------------------------------------------------------------------ drawing
    def annotate(self, frame: np.ndarray, tracks: list[dict]) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = self.s.stream_width / w
        img = cv2.resize(frame, (self.s.stream_width, int(h * scale)))
        overlay = img.copy()
        for zone in self.cfg.no_parking_zones:
            cv2.fillPoly(overlay, [(np.array(zone) * scale).astype(np.int32)], (0, 0, 200))
        img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
        for z in self.cfg.direction_zones:
            cv2.polylines(img, [(np.array(z.polygon) * scale).astype(np.int32)], True, (200, 200, 200), 1)
        if self.cfg.count_line:
            a, b = (np.array(self.cfg.count_line) * scale).astype(int)
            cv2.line(img, tuple(a), tuple(b), (0, 215, 255), 2)
        for tr in tracks:
            x1, y1, x2, y2 = (np.array(tr["box"]) * scale).astype(int)
            st = self.tracks.get(tr["track_id"])
            color = (0, 0, 255) if st and (st.wrong_way_reported or st.parking_reported) else \
                (0, 165, 255) if st and st.speeding_reported else CLASS_COLORS.get(tr["cls"], (200, 200, 200))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"#{tr['track_id']} {tr['cls']}"
            if tr["speed"]:
                label += f" {tr['speed']:.0f}km/h"
            if tr["plate"]:
                label += f" {tr['plate']}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            if st and len(st.image_pts) > 1:
                pts = (np.array([[p[1], p[2]] for p in st.image_pts]) * scale).astype(np.int32)
                cv2.polylines(img, [pts], False, color, 1)
        hud = f"{self.cfg.name} | vehicles {self.stats.passed} | {self.stats.fps:.1f} FPS"
        cv2.rectangle(img, (0, 0), (len(hud) * 9 + 12, 26), (20, 20, 20), -1)
        cv2.putText(img, hud, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return img
