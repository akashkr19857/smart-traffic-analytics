"""ANPR evaluation on the OpenALPR end-to-end benchmark (US + EU images with ground-truth plate box and text).

    python backend/eval/eval_anpr.py              # detection + OCR ablation
    python backend/eval/eval_anpr.py --ocr-only   # OCR on ground-truth crops only (no detector needed)

Reports
* plate detection recall / precision (IoU >= 0.5) of the trained detector on images it has never seen
* OCR exact-match accuracy on ground-truth crops: raw crop vs enhanced (upscale + CLAHE + deskew)
* multi-frame majority voting: each plate is re-captured 5 times with realistic degradations (scale, blur, brightness,
  rotation, JPEG) to SIMULATE consecutive video frames, then the reads are voted
* end-to-end: detector crop -> enhanced OCR -> exact match
Results: backend/eval/results/anpr.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT.parent / "_shared"))

from app.anpr.ocr import PlateOCR, PlateRead, normalise, vote  # noqa: E402

BENCH = ROOT / "data" / "raw" / "openalpr-benchmarks" / "endtoend"


def load(region: str) -> list[dict]:
    out = []
    for txt in sorted((BENCH / region).glob("*.txt")):
        parts = txt.read_text(encoding="utf-8", errors="ignore").strip().split()
        if len(parts) < 6:
            continue
        img, x, y, w, h, plate = parts[0], *map(int, parts[1:5]), parts[5]
        out.append({"region": region, "image": str(BENCH / region / img), "box": (x, y, w, h), "plate": normalise(plate)})
    return out


def degrade(crop: np.ndarray, rng: random.Random) -> np.ndarray:
    """One simulated video frame of the same plate."""
    h, w = crop.shape[:2]
    s = rng.uniform(0.55, 1.0)
    img = cv2.resize(crop, (max(8, int(w * s)), max(6, int(h * s))), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.6:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.7, 1.3), beta=rng.uniform(-25, 25))
    ang = rng.uniform(-6, 6)
    m = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), ang, 1.0)
    img = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(35, 80)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1, x2, y2 = max(ax, bx), max(ay, by), min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-only", action="store_true")
    ap.add_argument("--frames", type=int, default=5)
    args = ap.parse_args()
    items = load("us") + load("eu")
    ocr = PlateOCR()
    rng = random.Random(0)
    rows = []
    t_ocr = []
    for it in items:
        img = cv2.imread(it["image"])
        x, y, w, h = it["box"]
        crop = img[max(0, y):y + h, max(0, x):x + w]
        t0 = time.perf_counter()
        raw = ocr.read(crop, preprocess=False)
        t_ocr.append((time.perf_counter() - t0) * 1000)
        enh = ocr.read(crop, preprocess=True)
        frames = [degrade(crop, rng) for _ in range(args.frames)]
        single = [ocr.read(f, preprocess=True) for f in frames]
        voted = vote(single) or PlateRead("", 0.0)
        rows.append({"region": it["region"], "plate": it["plate"], "raw": raw.text, "enhanced": enh.text,
                     "frames": [r.text for r in single], "voted": voted.text})
    res = {"images": len(items), "us": sum(r["region"] == "us" for r in rows), "eu": sum(r["region"] == "eu" for r in rows)}

    def acc(key: str, region: str | None = None) -> float:
        sel = [r for r in rows if region is None or r["region"] == region]
        return round(sum(r[key] == r["plate"] for r in sel) / len(sel), 4)

    res["ocr_gt_crops"] = {
        "raw": {"all": acc("raw"), "us": acc("raw", "us"), "eu": acc("eu" and "raw", "eu")},
        "enhanced": {"all": acc("enhanced"), "us": acc("enhanced", "us"), "eu": acc("enhanced", "eu")},
        "ocr_ms_per_plate": round(float(np.median(t_ocr)), 1),
    }
    single_acc = sum(f == r["plate"] for r in rows for f in r["frames"]) / (len(rows) * args.frames)
    res["simulated_video"] = {"frames_per_plate": args.frames, "single_frame_accuracy": round(single_acc, 4),
                              "voted_accuracy": acc("voted"),
                              "note": "frames are simulated degradations of the benchmark still (scale, blur, brightness, rotation, JPEG)"}
    if not args.ocr_only:
        from ultralytics import YOLO

        det = YOLO(str(ROOT / "models" / "plate_yolo11n.pt"))
        tp = fp = fn = 0
        e2e = 0
        det_ms = []
        for it, row in zip(items, rows):
            img = cv2.imread(it["image"])
            t0 = time.perf_counter()
            r = det.predict(img, conf=0.35, imgsz=640, verbose=False)[0]
            det_ms.append((time.perf_counter() - t0) * 1000)
            boxes = [(int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])) for b in r.boxes.xyxy.cpu().numpy()]
            matched = [b for b in boxes if iou(b, it["box"]) >= 0.5]
            tp += bool(matched)
            fn += not matched
            fp += len(boxes) - len(matched[:1])
            if matched:
                bx, by, bw, bh = matched[0]
                pad = int(0.08 * bw)
                crop = img[max(0, by - pad):by + bh + pad, max(0, bx - pad):bx + bw + pad]
                text = ocr.read(crop, preprocess=True).text
                row["detector_read"] = text
                e2e += text == it["plate"]
        res["detection"] = {"recall_iou50": round(tp / len(items), 4), "precision_iou50": round(tp / max(tp + fp, 1), 4),
                            "detector_ms_per_image": round(float(np.median(det_ms[1:])), 1)}
        res["end_to_end_exact_match"] = round(e2e / len(items), 4)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / ("anpr_ocr_only.json" if args.ocr_only else "anpr.json")).write_text(json.dumps({"metrics": res, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
