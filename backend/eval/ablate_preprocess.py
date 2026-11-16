"""Ablation of each OCR preprocessing step on OpenALPR ground-truth crops, clean and degraded.

    python backend/eval/ablate_preprocess.py     → backend/eval/results/preprocess_ablation.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(HERE))

from app.anpr import ocr as O  # noqa: E402
from eval_anpr import degrade, load  # noqa: E402


def variant(crop: np.ndarray, gray: bool, upscale: int | None, use_clahe: bool, use_deskew: bool, pad: bool) -> np.ndarray:
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if gray or use_clahe or use_deskew else crop
    if upscale:
        s = upscale / max(img.shape[0], 1)
        if s > 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    if use_clahe:
        img = O.clahe(img)
    if use_deskew:
        img = O.deskew(img, O.skew_angle(img))
    if pad:
        img = cv2.copyMakeBorder(img, 6, 6, 10, 10, cv2.BORDER_REPLICATE)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


VARIANTS = {
    "raw": dict(gray=False, upscale=None, use_clahe=False, use_deskew=False, pad=False),
    "pad": dict(gray=False, upscale=None, use_clahe=False, use_deskew=False, pad=True),
    "gray": dict(gray=True, upscale=None, use_clahe=False, use_deskew=False, pad=False),
    "upscale48": dict(gray=False, upscale=48, use_clahe=False, use_deskew=False, pad=False),
    "upscale96": dict(gray=False, upscale=96, use_clahe=False, use_deskew=False, pad=False),
    "clahe": dict(gray=True, upscale=None, use_clahe=True, use_deskew=False, pad=False),
    "deskew": dict(gray=True, upscale=None, use_clahe=False, use_deskew=True, pad=False),
    "clahe+deskew": dict(gray=True, upscale=None, use_clahe=True, use_deskew=True, pad=False),
    "up48+clahe+deskew+pad": dict(gray=True, upscale=48, use_clahe=True, use_deskew=True, pad=True),
    "up96+clahe+deskew+pad (v1)": dict(gray=True, upscale=96, use_clahe=True, use_deskew=True, pad=True),
}


def main() -> None:
    items = load("us") + load("eu")
    eng = O.PlateOCR()
    rng = random.Random(1)
    crops = []
    for it in items:
        img = cv2.imread(it["image"])
        x, y, w, h = it["box"]
        crop = img[max(0, y):y + h, max(0, x):x + w]
        crops.append((it["plate"], crop, degrade(crop, rng)))
    res = {}
    for name, kw in VARIANTS.items():
        clean = sum(O.normalise(eng._rec(variant(c, **kw))[0]) == p for p, c, _ in crops) / len(crops)
        deg = sum(O.normalise(eng._rec(variant(d, **kw))[0]) == p for p, _, d in crops) / len(crops)
        res[name] = {"clean": round(clean, 4), "degraded": round(deg, 4)}
        print(f"{name:28s} clean {clean:.3f}  degraded {deg:.3f}", flush=True)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "preprocess_ablation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__" and "--strategies" not in sys.argv:
    main()


def strategies() -> None:
    """Adaptive alternatives to blanket preprocessing."""
    items = load("us") + load("eu")
    eng = O.PlateOCR()
    rng = random.Random(1)
    rows = []
    for it in items:
        img = cv2.imread(it["image"])
        x, y, w, h = it["box"]
        crop = img[max(0, y):y + h, max(0, x):x + w]
        rows.append((it["plate"], crop, degrade(crop, rng)))
    kw_cd = VARIANTS["clahe+deskew"]
    res = {}
    for label, idx in (("clean", 1), ("degraded", 2)):
        stats = {k: 0 for k in ("raw", "adaptive_clahe_std<35", "adaptive_clahe_std<45", "best_conf(raw,clahe+deskew)",
                                "best_conf(raw,deskew)")}
        for row in rows:
            p, c = row[0], row[idx]
            g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
            r_raw = eng._rec(c)
            r_cd = eng._rec(variant(c, **kw_cd))
            r_d = eng._rec(variant(c, **VARIANTS["deskew"]))
            std = float(g.std())
            stats["raw"] += O.normalise(r_raw[0]) == p
            stats["adaptive_clahe_std<35"] += O.normalise((r_cd if std < 35 else r_raw)[0]) == p
            stats["adaptive_clahe_std<45"] += O.normalise((r_cd if std < 45 else r_raw)[0]) == p
            stats["best_conf(raw,clahe+deskew)"] += O.normalise(max(r_raw, r_cd, key=lambda t: t[1])[0]) == p
            stats["best_conf(raw,deskew)"] += O.normalise(max(r_raw, r_d, key=lambda t: t[1])[0]) == p
        res[label] = {k: round(v / len(rows), 4) for k, v in stats.items()}
        print(label, res[label], flush=True)
    (HERE / "results" / "preprocess_strategies.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__" and "--strategies" in sys.argv:
    strategies()
