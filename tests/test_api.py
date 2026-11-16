from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import load_cameras
from app.db import Event, Vehicle, session
from app.main import app
from app.runner import CameraWorker
from app.vision.geometry import TrackState


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _worker() -> CameraWorker:
    return CameraWorker(load_cameras()[0], models=None)  # type: ignore[arg-type]


def test_health_and_cameras(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["models_loaded"] is False  # autostart disabled in tests
    cams = client.get("/api/cameras").json()
    assert [c["id"] for c in cams] == ["highway-d1", "highway-incidents", "expressway-city"]
    assert set(cams[0]["features"]) == {"speed", "wrong-way", "parking", "counting", "anpr"}
    assert client.get("/api/cameras/nope/snapshot.jpg").status_code == 404


def test_watchlist_crud_and_validation(client):
    r = client.post("/api/watchlist", json={"plate": "nz 412-gc", "reason": "stolen"})
    assert r.status_code == 200 and r.json()["plate"] == "NZ412GC"
    assert any(w["plate"] == "NZ412GC" for w in client.get("/api/watchlist").json())
    assert client.post("/api/watchlist", json={"plate": "-- ", "reason": ""}).status_code in (400, 422)
    assert client.delete("/api/watchlist/NZ412GC").status_code == 200
    assert client.delete("/api/watchlist/NZ412GC").status_code == 404


def test_event_persistence_snapshot_and_watchlist_hit(client):
    client.post("/api/watchlist", json={"plate": "ABC123", "reason": "unpaid fines"})
    w = _worker()
    frame = np.zeros((2160, 3840, 3), np.uint8)
    crop = np.full((40, 120, 3), 200, np.uint8)
    w._on_event({"type": "plate", "severity": "info", "track_id": 7, "cls": "car", "plate": "ABC123", "plate_conf": 0.91,
                 "speed_kmh": 88.0, "message": "car plate ABC123", "video_t": 3.2, "box": [100, 100, 400, 300], "zone": ""}, frame, crop)
    evs = client.get("/api/events?plate=abc").json()
    types = {e["type"] for e in evs}
    assert {"plate", "watchlist"} <= types
    plate_ev = next(e for e in evs if e["type"] == "plate")
    assert plate_ev["plate_crop"].endswith("_plate.jpg")
    img = client.get(f"/api/snapshots/{plate_ev['snapshot']}")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert client.get("/api/snapshots/..%2F..%2Fcameras.json").status_code == 404  # no path traversal
    found = client.get("/api/plates/search?q=bc12").json()
    assert found["query"] == "BC12" and found["sightings"][0]["plate"] == "ABC123"
    assert client.get("/api/events?type=watchlist").json()[0]["severity"] == "critical"


def test_vehicle_flush_and_analytics(client):
    w = _worker()
    from app.vision.pipeline import CameraPipeline

    w.pipeline = CameraPipeline.__new__(CameraPipeline)
    st = TrackState(11, "truck", 1.0, 3.0)
    for i in range(10):
        st.image_pts.append((1 + i * 0.2, 1000, 1200 + i * 20))
    st.speeds = [80.0, 82.0, 84.0]
    st.plate, st.counted = "XY987", True
    w.pipeline.finished, w.pipeline.tracks = [st], {}
    w._flush_tracks()
    with session() as db:
        v = db.query(Vehicle).filter(Vehicle.track_id == 11).one()
    assert v.cls == "truck" and v.avg_speed_kmh == 82.0 and v.direction == "toward camera" and v.plate == "XY987"
    a = client.get("/api/analytics").json()
    assert a["vehicles"] >= 1 and a["by_class"]["truck"] >= 1 and a["plates_read"] >= 1
    assert client.get("/api/vehicles?cls=truck").json()[0]["plate"] == "XY987"


def test_recognize_requires_models(client):
    files = {"file": ("x.jpg", b"not an image", "image/jpeg")}
    assert client.post("/api/anpr/recognize", files=files).status_code == 400


def test_benchmarks_endpoint(client):
    assert isinstance(client.get("/api/benchmarks").json(), dict)


def test_event_model_roundtrip():
    with session() as db:
        e = Event(camera_id="c", type="speeding", track_id=1, message="m", snapshot="s.jpg")
        db.add(e)
        db.commit()
        d = e.to_dict()
    assert d["type"] == "speeding" and "created_at" in d
