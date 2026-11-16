"""Background camera workers: read a video file / RTSP stream, run the pipeline, persist events and passages, publish
live events and the latest annotated JPEG for the MJPEG stream. One shared model set on one GPU; cameras take turns."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np

from .config import PROJECT_ROOT, CameraConfig, get_settings, load_cameras
from .db import Event, Vehicle, Watchlist, session
from .events import bus
from .vision.pipeline import CameraPipeline, Models

log = logging.getLogger("traffic.runner")
_gpu = threading.Lock()  # one camera at a time on the GPU


class CameraWorker:
    def __init__(self, cfg: CameraConfig, models: Models) -> None:
        self.cfg, self.models = cfg, models
        self.run_id = uuid.uuid4().hex[:8]
        self.jpeg: bytes | None = None
        self.jpeg_seq = 0
        self.status = "stopped"
        self.error = ""
        self.video_t = 0.0
        self.source_fps = 25.0
        self.resolution = (0, 0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.pipeline: CameraPipeline | None = None
        self.last_timings: dict[str, float] = {}

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"cam-{self.cfg.id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _open(self) -> cv2.VideoCapture:
        src = self.cfg.source
        path = src if "://" in src else str(PROJECT_ROOT / src)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open source {src}")
        self.source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return cap

    def _run(self) -> None:
        s = get_settings()
        try:
            cap = self._open()
        except Exception as exc:
            self.status, self.error = "error", str(exc)
            log.exception("camera %s failed to open", self.cfg.id)
            return
        self.pipeline = CameraPipeline(self.cfg, self.models, self._on_event, self.source_fps)
        self.status = "running"
        frame_idx = 0
        wall0 = time.perf_counter()
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                self._flush_tracks(final=True)
                if not self.cfg.loop:
                    break
                cap.release()
                cap = self._open()
                self.run_id = uuid.uuid4().hex[:8]
                self.pipeline = CameraPipeline(self.cfg, self.models, self._on_event, self.source_fps)
                frame_idx, wall0 = 0, time.perf_counter()
                continue
            t = frame_idx / self.source_fps
            self.video_t = t
            frame_idx += 1
            with _gpu:
                res = self.pipeline.process(frame, t)
            self.last_timings = res.timings_ms
            if res.annotated is not None:
                ok2, buf = cv2.imencode(".jpg", res.annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok2:
                    self.jpeg, self.jpeg_seq = buf.tobytes(), self.jpeg_seq + 1
            if frame_idx % 25 == 0:
                self._flush_tracks()
                bus.publish({"type": "stats", "camera_id": self.cfg.id, "stats": self.stats()})
            if s.realtime:  # pace to the video's own frame rate (skip ahead if we fall behind)
                ahead = t - (time.perf_counter() - wall0)
                if ahead > 0:
                    time.sleep(ahead)
                elif ahead < -1.0:
                    skip = int(-ahead * self.source_fps)
                    for _ in range(min(skip, 10)):
                        cap.grab()
                        frame_idx += 1
        cap.release()
        self.status = "stopped"

    # ------------------------------------------------------------------ persistence
    def _on_event(self, e: dict, frame: np.ndarray, crop: np.ndarray | None) -> None:
        s = get_settings()
        name = f"{self.cfg.id}_{self.run_id}_{e['track_id']}_{e['type']}_{int(time.time() * 1000)}"
        x1, y1, x2, y2 = map(int, e["box"])
        snap = frame.copy()
        cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 0, 255), 4)
        h, w = snap.shape[:2]
        scale = min(1.0, 1280 / w)
        snap = cv2.resize(snap, (int(w * scale), int(h * scale)))
        cv2.imwrite(str(s.snapshot_dir / f"{name}.jpg"), snap, [cv2.IMWRITE_JPEG_QUALITY, 80])
        crop_name = ""
        if crop is not None and crop.size:
            crop_name = f"{name}_plate.jpg"
            cv2.imwrite(str(s.snapshot_dir / crop_name), crop)
        with session() as db:
            ev = Event(camera_id=self.cfg.id, type=e["type"], severity=e["severity"], track_id=e["track_id"], cls=e["cls"],
                       plate=e.get("plate", ""), plate_conf=e.get("plate_conf", 0.0), speed_kmh=e.get("speed_kmh"),
                       message=e["message"], snapshot=f"{name}.jpg", plate_crop=crop_name, video_t=round(e["video_t"], 2),
                       details={"zone": e.get("zone", ""), "run_id": self.run_id})
            db.add(ev)
            hit = db.get(Watchlist, e["plate"]) if e["type"] == "plate" and e.get("plate") else None
            if hit is not None:
                db.add(Event(camera_id=self.cfg.id, type="watchlist", severity="critical", track_id=e["track_id"], cls=e["cls"],
                             plate=e["plate"], plate_conf=e.get("plate_conf", 0.0), message=f"Watchlist plate {e['plate']}: {hit.reason}",
                             snapshot=f"{name}.jpg", plate_crop=crop_name, video_t=round(e["video_t"], 2)))
            db.commit()
            bus.publish({"type": "event", "event": ev.to_dict()})

    def _flush_tracks(self, final: bool = False) -> None:
        p = self.pipeline
        if p is None:
            return
        done = p.finished + (list(p.tracks.values()) if final else [])
        p.finished = []
        if final:
            p.tracks = {}
        rows = []
        vp = self.cfg.vanishing_point
        for st in done:
            if len(st.image_pts) < 5:
                continue
            direction = ""
            if vp and len(st.image_pts) >= 2:
                y0, y1 = st.image_pts[0][2], st.image_pts[-1][2]
                direction = "toward camera" if y1 > y0 else "away from camera"
            sp = sorted(st.speeds)
            rows.append(Vehicle(camera_id=self.cfg.id, track_id=st.track_id, run_id=self.run_id, cls=st.cls,
                                video_t=round(st.first_ts, 2), direction=direction,
                                avg_speed_kmh=round(sp[len(sp) // 2], 1) if sp else None, max_speed_kmh=round(max(sp), 1) if sp else None,
                                plate=st.plate, plate_conf=st.plate_conf, plate_reads=len(st.plate_reads), counted=st.counted,
                                first_seen=datetime.now(timezone.utc), last_seen=datetime.now(timezone.utc)))
        if rows:
            with session() as db:
                db.add_all(rows)
                db.commit()

    def stats(self) -> dict:
        p = self.pipeline
        if p is None:
            return {"status": self.status}
        sp = list(p.stats.speeds)
        return {"status": self.status, "fps": round(p.stats.fps, 1), "video_t": round(self.video_t, 1),
                "counts": dict(p.stats.counts), "passed": p.stats.passed, "active_tracks": len(p.tracks),
                "avg_speed_kmh": round(float(np.median(sp)), 1) if sp else None, "events": dict(p.stats.events),
                "timings_ms": {k: round(v, 1) for k, v in self.last_timings.items()}, "resolution": list(self.resolution)}


class Fleet:
    def __init__(self) -> None:
        self.models: Models | None = None
        self.workers: dict[str, CameraWorker] = {}

    def start(self) -> None:
        if self.models is None:
            self.models = Models.load(with_anpr=any(c.anpr for c in load_cameras()))
        for cfg in load_cameras():
            w = self.workers.get(cfg.id) or CameraWorker(cfg, self.models)
            self.workers[cfg.id] = w
            w.start()

    def stop(self) -> None:
        for w in self.workers.values():
            w.stop()


fleet = Fleet()
