"""AI Smart Traffic Analytics API (port 8400): live MJPEG streams, events, vehicles, ANPR search, watchlist, analytics."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func

from . import __version__
from .anpr.ocr import normalise
from .config import PROJECT_ROOT, get_settings, load_cameras
from .db import Event, Vehicle, Watchlist, session
from .events import bus
from .runner import fleet

log = logging.getLogger("traffic")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if get_settings().autostart:
        threading.Thread(target=fleet.start, daemon=True).start()
    yield
    fleet.stop()


app = FastAPI(title="AI Smart Traffic Analytics API", version=__version__, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins.split(","), allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health() -> dict:
    import torch

    return {"status": "ok", "version": __version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "models_loaded": fleet.models is not None, "cameras": {cid: w.status for cid, w in fleet.workers.items()}}


# ------------------------------------------------------------------ cameras & live video
@app.get("/api/cameras")
def cameras() -> list[dict]:
    out = []
    for c in load_cameras():
        w = fleet.workers.get(c.id)
        out.append({"id": c.id, "name": c.name, "description": c.description, "anpr": c.anpr,
                    "speed_limit_kmh": c.speed_limit_kmh, "features": [f for f, on in (("speed", c.speed_source), ("wrong-way", c.direction_zones),
                                                                                        ("parking", c.no_parking_zones), ("counting", c.count_line),
                                                                                        ("anpr", c.anpr)) if on],
                    "stats": w.stats() if w else {"status": "not started"}})
    return out


def _worker(cam_id: str):  # noqa: ANN202
    w = fleet.workers.get(cam_id)
    if w is None:
        raise HTTPException(404, "camera not running")
    return w


@app.get("/api/cameras/{cam_id}/stream.mjpg")
async def stream(cam_id: str) -> StreamingResponse:
    """Annotated live video as multipart MJPEG (works directly in an <img> tag)."""
    w = _worker(cam_id)

    async def gen():
        seq = -1
        while True:
            if w.jpeg is not None and w.jpeg_seq != seq:
                seq = w.jpeg_seq
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + w.jpeg + b"\r\n"
            await asyncio.sleep(0.03)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/cameras/{cam_id}/snapshot.jpg")
def camera_snapshot(cam_id: str) -> Response:
    w = _worker(cam_id)
    if w.jpeg is None:
        raise HTTPException(503, "no frame yet")
    return Response(w.jpeg, media_type="image/jpeg")


@app.post("/api/cameras/{cam_id}/restart")
def restart(cam_id: str) -> dict:
    w = _worker(cam_id)
    w.stop()
    time.sleep(0.3)
    w.start()
    return {"restarted": cam_id}


# ------------------------------------------------------------------ events & vehicles
@app.get("/api/events")
def events(camera_id: str | None = None, type: str | None = None, plate: str | None = None, limit: int = 100) -> list[dict]:
    with session() as db:
        q = db.query(Event)
        if camera_id:
            q = q.filter(Event.camera_id == camera_id)
        if type:
            q = q.filter(Event.type.in_(type.split(",")))
        if plate:
            q = q.filter(Event.plate.like(f"%{normalise(plate)}%"))
        return [e.to_dict() for e in q.order_by(Event.id.desc()).limit(min(limit, 500)).all()]


@app.get("/api/snapshots/{name}")
def snapshot_file(name: str) -> FileResponse:
    path = (get_settings().snapshot_dir / name).resolve()
    if path.parent != get_settings().snapshot_dir.resolve() or not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/vehicles")
def vehicles(camera_id: str | None = None, cls: str | None = None, limit: int = 200) -> list[dict]:
    with session() as db:
        q = db.query(Vehicle)
        if camera_id:
            q = q.filter(Vehicle.camera_id == camera_id)
        if cls:
            q = q.filter(Vehicle.cls == cls)
        return [v.to_dict() for v in q.order_by(Vehicle.id.desc()).limit(min(limit, 1000)).all()]


@app.get("/api/plates/search")
def plate_search(q: str = Query(min_length=2)) -> dict:
    """Every sighting of a (partial) plate across cameras, from both plate events and vehicle passages."""
    needle = normalise(q)
    with session() as db:
        evs = db.query(Event).filter(Event.type == "plate", Event.plate.like(f"%{needle}%")).order_by(Event.id.desc()).limit(100).all()
        vs = db.query(Vehicle).filter(Vehicle.plate.like(f"%{needle}%")).order_by(Vehicle.id.desc()).limit(100).all()
        return {"query": needle, "sightings": [e.to_dict() for e in evs], "passages": [v.to_dict() for v in vs]}


class WatchIn(BaseModel):
    plate: str = Field(min_length=3, max_length=16)
    reason: str = Field(default="", max_length=300)


@app.get("/api/watchlist")
def watchlist() -> list[dict]:
    with session() as db:
        return [w.to_dict() for w in db.query(Watchlist).order_by(Watchlist.created_at.desc()).all()]


@app.post("/api/watchlist")
def watch_add(body: WatchIn) -> dict:
    plate = normalise(body.plate)
    if len(plate) < 3:
        raise HTTPException(400, "plate must have at least 3 letters/digits")
    with session() as db:
        w = db.get(Watchlist, plate) or Watchlist(plate=plate)
        w.reason = body.reason
        db.merge(w)
        db.commit()
        return w.to_dict()


@app.delete("/api/watchlist/{plate}")
def watch_remove(plate: str) -> dict:
    with session() as db:
        w = db.get(Watchlist, normalise(plate))
        if w is None:
            raise HTTPException(404, "not on watchlist")
        db.delete(w)
        db.commit()
    return {"removed": normalise(plate)}


# ------------------------------------------------------------------ ANPR on an uploaded image
@app.post("/api/anpr/recognize")
async def recognize(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "not an image")
    if fleet.models is None or fleet.models.plate is None:
        raise HTTPException(503, "plate model not loaded")
    t0 = time.perf_counter()
    r = fleet.models.plate.predict(img, conf=get_settings().plate_conf, imgsz=640, verbose=False)[0]
    plates = []
    for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = box.astype(int)
        pad = int(0.06 * (x2 - x1))
        read = fleet.models.ocr.read(img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad])
        plates.append({"box": [int(x1), int(y1), int(x2), int(y2)], "det_conf": round(float(conf), 3), "text": read.text,
                       "ocr_conf": round(read.conf, 3)})
    return {"plates": plates, "ms": round((time.perf_counter() - t0) * 1000, 1)}


# ------------------------------------------------------------------ analytics
@app.get("/api/analytics")
def analytics(camera_id: str | None = None, minutes: int = 120) -> dict:
    since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    with session() as db:
        vq = db.query(Vehicle).filter(Vehicle.first_seen >= since)
        eq = db.query(Event).filter(Event.created_at >= since)
        if camera_id:
            vq, eq = vq.filter(Vehicle.camera_id == camera_id), eq.filter(Event.camera_id == camera_id)
        vs, evs = vq.all(), eq.all()
        total_plates = db.query(func.count(func.distinct(Vehicle.plate))).filter(Vehicle.plate != "").scalar()
    by_min: dict[str, Counter] = {}
    for v in vs:
        key = v.first_seen.replace(second=0, microsecond=0).strftime("%H:%M")
        by_min.setdefault(key, Counter())[v.cls] += 1
    speeds = [v.avg_speed_kmh for v in vs if v.avg_speed_kmh]
    hist = Counter(int(s // 10 * 10) for s in speeds)
    return {"vehicles": len(vs), "by_class": dict(Counter(v.cls for v in vs)),
            "per_minute": [{"minute": k, **dict(c)} for k, c in sorted(by_min.items())],
            "speed_histogram": [{"bucket": f"{b}-{b + 9}", "count": hist[b]} for b in sorted(hist)],
            "avg_speed_kmh": round(float(np.median(speeds)), 1) if speeds else None,
            "events_by_type": dict(Counter(e.type for e in evs)), "distinct_plates": total_plates,
            "plates_read": sum(1 for v in vs if v.plate)}


@app.get("/api/benchmarks")
def benchmarks() -> dict:
    res = PROJECT_ROOT / "backend" / "eval" / "results"
    out = {}
    for name in ("anpr", "rules", "preprocess_ablation", "preprocess_strategies", "fps"):
        f = res / f"{name}.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            out[name] = d.get("metrics", d) if name == "anpr" else d
    det = PROJECT_ROOT / "models" / "plate_detector_metrics.json"
    if det.exists():
        out["plate_detector"] = json.loads(det.read_text(encoding="utf-8"))
    return out


@app.websocket("/ws/events")
async def ws_events(ws: WebSocket) -> None:
    await ws.accept()
    q = bus.subscribe()
    try:
        while True:
            await ws.send_json(json.loads(json.dumps(await q.get(), default=str)))
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(q)
