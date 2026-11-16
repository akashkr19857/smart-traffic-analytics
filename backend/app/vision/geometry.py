"""Pure geometry for speed estimation and traffic rules (no model code, fully unit-tested)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


class ViewTransformer:
    """Perspective transform from image pixels to a bird's-eye plane in metres (homography from 4 point pairs)."""

    def __init__(self, source: list[tuple[float, float]], target_wh_m: tuple[float, float]) -> None:
        w, h = target_wh_m
        src = np.array(source, dtype=np.float32)
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        self.m = cv2.getPerspectiveTransform(src, dst)

    def transform(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return pts.reshape(-1, 2)
        return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), self.m).reshape(-1, 2)


def point_in_polygon(pt: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    return cv2.pointPolygonTest(np.array(poly, dtype=np.float32), (float(pt[0]), float(pt[1])), False) >= 0


def side_of_line(pt: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])


@dataclass
class TrackState:
    """Everything the rules need to know about one tracked vehicle."""
    track_id: int
    cls: str
    first_ts: float
    last_ts: float
    image_pts: deque = field(default_factory=lambda: deque(maxlen=90))  # (t, x, y) bottom-centre, full-res px
    world_pts: deque = field(default_factory=lambda: deque(maxlen=90))  # (t, X, Y) metres
    speeds: list[float] = field(default_factory=list)
    counted: bool = False
    wrong_way_frames: int = 0
    wrong_way_reported: bool = False
    stationary_since: float | None = None
    parking_reported: bool = False
    speeding_reported: bool = False
    zone: str = ""
    plate_reads: list = field(default_factory=list)
    plate: str = ""
    plate_conf: float = 0.0
    box_h: float = 0.0


def speed_kmh(world_pts: deque, window_s: float = 1.0) -> float | None:
    """Speed from the bird's-eye trajectory over (at least) the last `window_s` seconds."""
    if len(world_pts) < 3:
        return None
    t1, x1, y1 = world_pts[-1]
    start = None
    for t, x, y in reversed(world_pts):
        if t1 - t >= window_s:
            start = (t, x, y)
            break
    if start is None:
        return None
    t0, x0, y0 = start
    dist = float(np.hypot(x1 - x0, y1 - y0))
    return dist / max(t1 - t0, 1e-6) * 3.6


def heading_sign(image_pts: deque, vp: tuple[float, float], min_move_px: float = 25.0) -> int:
    """+1 if the vehicle moves away from the vanishing point (toward the camera), -1 toward it, 0 if unclear."""
    if len(image_pts) < 5:
        return 0
    _, x0, y0 = image_pts[0]
    _, x1, y1 = image_pts[-1]
    move = np.array([x1 - x0, y1 - y0])
    if np.linalg.norm(move) < min_move_px:
        return 0
    out = np.array([x1 - vp[0], y1 - vp[1]], dtype=float)
    out /= np.linalg.norm(out) + 1e-9
    d = float(move @ out) / (np.linalg.norm(move) + 1e-9)
    return 1 if d > 0.5 else -1 if d < -0.5 else 0


def displacement_px(image_pts: deque, since_ts: float) -> float:
    pts = [(x, y) for t, x, y in image_pts if t >= since_ts]
    if len(pts) < 2:
        return 0.0
    arr = np.array(pts)
    return float(np.max(np.linalg.norm(arr - arr.mean(axis=0), axis=1)))
