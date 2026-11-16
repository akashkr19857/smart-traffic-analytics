"""ByteTrack multi-object tracking (Ultralytics' implementation of Zhang et al., 2022) behind a small adapter that takes
and returns `supervision.Detections`."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import supervision as sv
from ultralytics.trackers.byte_tracker import BYTETracker


class _Boxes:
    """Minimal stand-in for ultralytics' Boxes: what BYTETracker reads (conf, cls, xywh, xyxy) plus indexing."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = xyxy.astype(np.float32)
        self.conf = conf.astype(np.float32)
        self.cls = cls.astype(np.float32)
        wh = self.xyxy[:, 2:4] - self.xyxy[:, 0:2]
        self.xywh = np.concatenate([self.xyxy[:, 0:2] + wh / 2, wh], axis=1) if len(self.xyxy) else np.zeros((0, 4), np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx):  # noqa: ANN001, ANN204
        return _Boxes(self.xyxy[idx], self.conf[idx], self.cls[idx])


class ByteTrack:
    """ByteTrack: two-stage association - high-confidence detections first, then low-confidence ones to keep tracks
    alive through occlusion. `lost_track_buffer` frames of memory before a track is dropped."""

    def __init__(self, frame_rate: int = 25, lost_track_buffer: int = 30, high_thresh: float = 0.4, low_thresh: float = 0.1,
                 new_track_thresh: float = 0.45, match_thresh: float = 0.8) -> None:
        args = SimpleNamespace(track_high_thresh=high_thresh, track_low_thresh=low_thresh, new_track_thresh=new_track_thresh,
                               track_buffer=lost_track_buffer, match_thresh=match_thresh, fuse_score=True)
        self.frame_rate = frame_rate  # informational: the buffer is already expressed in frames
        self.tracker = BYTETracker(args)

    def update_with_detections(self, det: sv.Detections) -> sv.Detections:
        boxes = _Boxes(det.xyxy if len(det) else np.zeros((0, 4)), det.confidence if len(det) else np.zeros(0),
                       det.class_id if len(det) else np.zeros(0))
        out = self.tracker.update(boxes)
        if out is None or len(out) == 0:
            return sv.Detections.empty() if hasattr(sv.Detections, "empty") else sv.Detections(xyxy=np.zeros((0, 4)))
        out = np.asarray(out)
        res = sv.Detections(xyxy=out[:, 0:4].astype(np.float32), confidence=out[:, 5].astype(np.float32),
                            class_id=out[:, 6].astype(int))
        res.tracker_id = out[:, 4].astype(int)
        return res
