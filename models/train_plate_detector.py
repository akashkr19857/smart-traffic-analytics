"""Train the license-plate detector (YOLO11n) on the CC BY 4.0 plate dataset, then evaluate on its test split.

    python models/train_plate_detector.py --epochs 30

Writes models/plate_yolo11n.pt and models/plate_detector_metrics.json (mAP on the held-out test split).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
from contextlib import nullcontext as gpu_lock  # single-process: no cross-process GPU lock needed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--base", default="yolo11n.pt")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    from ultralytics import YOLO

    data = ROOT / "data" / "plates_yolo" / "data.yaml"
    t0 = time.time()
    with gpu_lock("traffic-plate-training"):
        model = YOLO(str(HERE / args.base) if (HERE / args.base).exists() else args.base)
        model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers, device=0,
                    project=str(HERE / "runs"), name="plates", exist_ok=True, patience=10, seed=0, plots=True, verbose=False)
        best = HERE / "runs" / "plates" / "weights" / "best.pt"
        shutil.copy2(best, HERE / "plate_yolo11n.pt")
        test = YOLO(str(HERE / "plate_yolo11n.pt")).val(data=str(data), split="test", imgsz=args.imgsz, batch=args.batch,
                                                       device=0, workers=args.workers, plots=False, verbose=False)
    metrics = {"model": "YOLO11n fine-tuned", "epochs": args.epochs, "imgsz": args.imgsz,
               "train_minutes": round((time.time() - t0) / 60, 1),
               "test_split": {"images": 882, "mAP50": round(float(test.box.map50), 4), "mAP50_95": round(float(test.box.map), 4),
                              "precision": round(float(test.box.mp), 4), "recall": round(float(test.box.mr), 4)}}
    (HERE / "plate_detector_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
