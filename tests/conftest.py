"""Isolated temp data dir, no camera autostart, stub models (no GPU / no model download needed)."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
_TMP = Path(tempfile.mkdtemp(prefix="traffic-test-"))
os.environ.update({"TRAFFIC_DATA_DIR": str(_TMP), "TRAFFIC_AUTOSTART": "false", "TRAFFIC_DEVICE": "cpu"})

import numpy as np  # noqa: E402
import pytest  # noqa: E402


class _Boxes:
    def __init__(self, xyxy, conf, cls):  # noqa: ANN001
        import torch

        self.xyxy, self.conf, self.cls = torch.tensor(xyxy, dtype=torch.float32), torch.tensor(conf), torch.tensor(cls)

    def __len__(self) -> int:
        return len(self.conf)


class StubYOLO:
    """Mimics ultralytics YOLO.predict: returns boxes from a callable(frame_index) -> list[(x1,y1,x2,y2,cls)]."""

    def __init__(self, script):  # noqa: ANN001
        self.script, self.calls = script, 0
        self.names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 0: "license_plate"}

    def predict(self, img, **kw):  # noqa: ANN001, ANN003
        boxes = self.script(self.calls, img)
        self.calls += 1
        xyxy = [b[:4] for b in boxes] or np.zeros((0, 4))
        r = type("R", (), {})()
        r.boxes = _Boxes(xyxy, [0.9] * len(boxes), [b[4] for b in boxes])
        r.names = self.names
        r.orig_shape = img.shape[:2]
        return [r]


@pytest.fixture()
def stub_yolo():
    return StubYOLO
