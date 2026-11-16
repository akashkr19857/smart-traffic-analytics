from __future__ import annotations

from collections import deque

import cv2
import numpy as np
import pytest
import supervision as sv

from app.anpr.ocr import PlateRead, candidates, normalise, skew_angle, vote
from app.config import CameraConfig, DirectionZone, load_cameras
from app.vision.geometry import ViewTransformer, displacement_px, heading_sign, point_in_polygon, side_of_line, speed_kmh
from app.vision.tracking import ByteTrack


# ---------------------------------------------------------------- geometry & speed
def test_view_transformer_maps_quad_to_metres():
    vt = ViewTransformer([(100, 100), (300, 100), (400, 500), (0, 500)], (10, 50))
    out = vt.transform(np.array([[100, 100], [400, 500], [200, 300]], dtype=float))
    assert np.allclose(out[0], [0, 0], atol=1e-3) and np.allclose(out[1], [10, 50], atol=1e-3)
    assert 0 < out[2][0] < 10 and 0 < out[2][1] < 50


def test_speed_from_world_trajectory():
    pts = deque((t / 25, 0.0, t / 25 * 25.0) for t in range(40))  # 25 m/s
    assert speed_kmh(pts) == pytest.approx(90.0, rel=1e-3)
    assert speed_kmh(deque([(0, 0, 0), (0.04, 0, 1)])) is None  # too short


def test_heading_relative_to_vanishing_point():
    vp = (500, 100)
    toward_camera = deque((i, 480 - i * 2, 300 + i * 20) for i in range(10))  # moving down, away from the VP
    away = deque((i, 480 + i * 2, 500 - i * 20) for i in range(10))
    still = deque((i, 480, 300) for i in range(10))
    assert heading_sign(toward_camera, vp) == 1 and heading_sign(away, vp) == -1 and heading_sign(still, vp) == 0


def test_polygon_line_and_displacement():
    poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), poly) and not point_in_polygon((15, 5), poly)
    assert side_of_line((0, -1), (0, 0), (10, 0)) * side_of_line((0, 1), (0, 0), (10, 0)) < 0
    pts = deque([(0, 100, 100), (1, 101, 100), (2, 100, 101), (3, 300, 300)])
    assert displacement_px(pts, 0) > 50 and displacement_px(pts, 0) > displacement_px(deque(list(pts)[:3]), 0)


def test_camera_config_file_is_valid():
    cams = load_cameras()
    ids = [c.id for c in cams]
    assert ids == ["highway-d1", "highway-incidents", "expressway-city"]
    hw = cams[0]
    assert hw.speed_source and hw.anpr and len(hw.direction_zones) == 2 and len(hw.no_parking_zones) == 2
    # the vanishing point must lie above both carriageway polygons
    assert all(min(y for _, y in z.polygon) > hw.vanishing_point[1] for z in hw.direction_zones)


# ---------------------------------------------------------------- tracking
def test_bytetrack_keeps_ids_through_motion():
    bt = ByteTrack(frame_rate=25, lost_track_buffer=30)
    ids = []
    for i in range(12):
        det = sv.Detections(xyxy=np.array([[100 + i * 5, 100, 160 + i * 5, 150], [400, 300 - i * 4, 470, 360 - i * 4]], np.float32),
                            confidence=np.array([0.9, 0.85], np.float32), class_id=np.array([2, 7]))
        out = bt.update_with_detections(det)
        ids.append(sorted(out.tracker_id.tolist()))
    assert len(set(map(tuple, ids[2:]))) == 1 and len(ids[-1]) == 2  # same two ids for the whole sequence


def test_bytetrack_empty_frames():
    bt = ByteTrack()
    out = bt.update_with_detections(sv.Detections(xyxy=np.zeros((0, 4), np.float32), confidence=np.zeros(0, np.float32),
                                                  class_id=np.zeros(0, int)))
    assert len(out) == 0


# ---------------------------------------------------------------- OCR helpers
def test_normalise_and_vote():
    assert normalise("yg9=x2-g ") == "YG9X2G"
    reads = [PlateRead("NZ412GC", 0.9), PlateRead("NZ412GG", 0.6), PlateRead("NZ4126C", 0.7), PlateRead("NZ412GC", 0.8),
             PlateRead("Z412GC", 0.95)]
    v = vote(reads)
    assert v.text == "NZ412GC" and 0 < v.conf <= 1
    assert vote([PlateRead("AB", 0.9)]) is None


def test_candidates_and_skew():
    img = np.full((60, 200, 3), 255, np.uint8)
    cv2.putText(img, "AB123", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    rot = cv2.warpAffine(img, cv2.getRotationMatrix2D((100, 30), 8, 1.0), (200, 60), borderValue=(255, 255, 255))
    assert abs(skew_angle(cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY))) > 3
    c = candidates(rot)
    assert len(c) == 3 and all(x.ndim == 3 for x in c) and c[0] is rot


# ---------------------------------------------------------------- pipeline with stub models
def _cfg(**kw) -> CameraConfig:
    base = dict(id="t", name="test", source="x.mp4", infer_width=640, speed_source=[(0, 0), (640, 0), (640, 480), (0, 480)],
                speed_target_m=(20, 100), speed_limit_kmh=100, vanishing_point=(320, -500),
                direction_zones=[DirectionZone(name="lane", polygon=[(0, 0), (640, 0), (640, 480), (0, 480)], legal="away")],
                no_parking_zones=[[(500, 300), (640, 300), (640, 480), (500, 480)]], parking_seconds=2.0,
                count_line=((0, 240), (640, 240)))
    base.update(kw)
    return CameraConfig(**base)


def _run(script, cfg: CameraConfig, frames: int, stub_yolo, fps: float = 10.0):  # noqa: ANN001
    from app.vision.pipeline import CameraPipeline, Models

    m = Models(stub_yolo(script), None, None)
    p = CameraPipeline(cfg, m, None, fps)
    events = []
    for i in range(frames):
        r = p.process(np.zeros((480, 640, 3), np.uint8), i / fps, annotate=(i == frames - 1))
        events += r.events
    return p, events, r


def test_pipeline_counts_and_measures_speed(stub_yolo):
    # car moves down 10 px/frame at 10 fps: 100 px/s; 480 px = 100 m → 20.8 m/s = 75 km/h
    p, events, last = _run(lambda i, img: [(300, 50 + i * 10, 360, 100 + i * 10, 2)], _cfg(), 30, stub_yolo)
    st = list(p.tracks.values())[0]
    assert p.stats.counts["car"] == 1 and st.counted
    assert np.median(st.speeds) == pytest.approx(75.0, rel=0.05)
    assert not [e for e in events if e["type"] in ("wrong_way", "speeding")]
    assert last.annotated is not None and last.annotated.shape[1] == 1280


def test_pipeline_flags_speeding(stub_yolo):
    _, events, _ = _run(lambda i, img: [(300, 20 + i * 18, 360, 70 + i * 18, 2)], _cfg(), 22, stub_yolo)  # ~135 km/h
    sp = [e for e in events if e["type"] == "speeding"]
    assert len(sp) == 1 and sp[0]["speed_kmh"] > 100


def test_pipeline_flags_wrong_way(stub_yolo):
    _, events, _ = _run(lambda i, img: [(300, 420 - i * 12, 360, 470 - i * 12, 2)], _cfg(), 30, stub_yolo)  # moving up = toward VP
    ww = [e for e in events if e["type"] == "wrong_way"]
    assert len(ww) == 1 and ww[0]["severity"] == "critical"


def test_pipeline_flags_illegal_parking_after_threshold(stub_yolo):
    _, events, _ = _run(lambda i, img: [(540, 380, 600, 430, 2)], _cfg(), 40, stub_yolo)  # 4 s stationary in the zone
    park = [e for e in events if e["type"] == "illegal_parking"]
    assert len(park) == 1 and park[0]["video_t"] >= 2.0
    _, events2, _ = _run(lambda i, img: [(540, 380, 600, 430, 2)], _cfg(), 15, stub_yolo)  # only 1.5 s
    assert not [e for e in events2 if e["type"] == "illegal_parking"]


def test_moving_vehicle_in_no_parking_zone_is_not_flagged(stub_yolo):
    _, events, _ = _run(lambda i, img: [(520, 300 + i * 4, 580, 350 + i * 4, 2)], _cfg(), 30, stub_yolo)
    assert not [e for e in events if e["type"] == "illegal_parking"]
