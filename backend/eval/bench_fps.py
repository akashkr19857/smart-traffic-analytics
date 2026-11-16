"""Inference speed on real 4K highway frames (vehicle detector only, then the full pipeline).

    python backend/eval/bench_fps.py   → backend/eval/results/fps.json
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from contextlib import nullcontext as gpu_lock  # single-process: no cross-process GPU lock needed


def frames(n: int = 120) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(ROOT / "data" / "raw" / "vehicles.mp4"))
    out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def bench(model, imgs: list[np.ndarray], imgsz: int, q: int, device: str) -> float:  # noqa: ANN001
    for f in imgs[:10]:
        model.predict(cv2.resize(f, (imgsz, imgsz * 9 // 16)), imgsz=imgsz, quantize=q, device=device, verbose=False)
    t = []
    for f in imgs[10:]:
        t0 = time.perf_counter()
        small = cv2.resize(f, (imgsz, imgsz * 9 // 16))
        model.predict(small, imgsz=imgsz, quantize=q, device=device, classes=[2, 3, 5, 7], verbose=False)
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))


def main() -> None:
    import torch
    from ultralytics import YOLO

    imgs = frames()
    rows = []
    with gpu_lock("traffic-fps-bench"):
        for name, imgsz, q, dev in (("yolo11s", 1280, 32, "cuda:0"), ("yolo11s", 1280, 16, "cuda:0"), ("yolo11s", 640, 16, "cuda:0"),
                                    ("yolo11n", 1280, 16, "cuda:0"), ("yolo11s", 1280, 32, "cpu")):
            m = YOLO(str(ROOT / "models" / f"{name}.pt"))
            sub = imgs if dev != "cpu" else imgs[:30]
            ms = bench(m, sub, imgsz, q, dev)
            rows.append({"config": f"{name} @{imgsz}px {'FP16' if q == 16 else 'FP32'} on {'GPU' if dev != 'cpu' else 'CPU'} (resize + inference)",
                         "ms_per_frame": round(ms, 1), "fps": round(1000 / ms, 1)})
            print(rows[-1], flush=True)
    hw = f"{torch.cuda.get_device_name(0)}, {platform.processor()}, {platform.system()} {platform.release()}; frames: vehicles.mp4 3840x2160"
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "fps.json").write_text(json.dumps({"hardware": hw, "rows": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
