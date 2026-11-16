"""Plate OCR: candidate preprocessing (CLAHE, deskew), RapidOCR recognition, normalisation and multi-frame voting.

Design follows measurements on 330 labelled OpenALPR plates (backend/eval/ablate_preprocess.py):
* blanket preprocessing hurt (padding -5.5 pts, CLAHE -2.8 pts) because the recogniser normalises its input itself;
* reading three candidates - raw, deskewed, CLAHE+deskewed - and keeping the most confident read helped slightly;
* the large gain comes from per-character, confidence-weighted voting over the reads of one tracked vehicle.
"""
from __future__ import annotations

import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

_ALNUM = re.compile(r"[^A-Z0-9]")


@dataclass
class PlateRead:
    text: str
    conf: float
    raw_text: str = ""


def normalise(text: str) -> str:
    return _ALNUM.sub("", (text or "").upper())


def clahe(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 8)).apply(gray)


def skew_angle(gray: np.ndarray) -> float:
    """Dominant text angle in degrees from the long edges of character blobs (minAreaRect of the text mask)."""
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    h, w = th.shape
    th[: int(h * 0.12)], th[int(h * 0.88):] = 0, 0  # ignore the plate border rows
    pts = cv2.findNonZero(th)
    if pts is None or len(pts) < 30:
        return 0.0
    (_, _), (rw, rh), ang = cv2.minAreaRect(pts)
    if rw < rh:
        ang -= 90
    if ang < -45:
        ang += 90
    if ang > 45:
        ang -= 90
    return float(ang) if abs(ang) <= 20 else 0.0


def deskew(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.7:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def candidates(crop: np.ndarray) -> list[np.ndarray]:
    """Raw crop, deskewed, and CLAHE + deskewed (grayscale) versions of a plate crop."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    ang = skew_angle(gray)
    desk = deskew(gray, ang)
    cl = deskew(clahe(gray), ang)
    return [crop, cv2.cvtColor(desk, cv2.COLOR_GRAY2BGR), cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)]


def enhance(crop: np.ndarray) -> np.ndarray:
    """CLAHE + deskew only (kept for the ablation and the API's debug view)."""
    return candidates(crop)[2]


class PlateOCR:
    def __init__(self) -> None:
        from rapidocr import RapidOCR

        self.engine = RapidOCR(params={"Global.log_level": "error"}) if _supports_params() else RapidOCR()
        self._lock = threading.Lock()

    def _rec(self, img: np.ndarray) -> tuple[str, float]:
        with self._lock:
            out = self.engine(img, use_det=False, use_cls=False, use_rec=True)
        txts, scores = getattr(out, "txts", None) or (), getattr(out, "scores", None) or ()
        return (txts[0], float(scores[0])) if txts else ("", 0.0)

    def _det_rec(self, img: np.ndarray) -> tuple[str, float]:
        """Two-line plates: detect text lines, read each, join top-to-bottom."""
        with self._lock:
            out = self.engine(img)
        txts, scores, boxes = getattr(out, "txts", None), getattr(out, "scores", None), getattr(out, "boxes", None)
        if not txts:
            return "", 0.0
        order = np.argsort([float(np.mean(np.asarray(b)[:, 1])) for b in boxes]) if boxes is not None else range(len(txts))
        return "".join(txts[i] for i in order), float(np.mean(scores))

    def read(self, crop: np.ndarray, preprocess: bool = True) -> PlateRead:
        """preprocess=True: best-confidence read over raw / deskew / CLAHE+deskew candidates; False: raw crop only."""
        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 8:
            return PlateRead("", 0.0)
        imgs = candidates(crop) if preprocess else [crop]
        best = ("", 0.0)
        for img in imgs:
            raw, conf = self._rec(img)
            if conf > best[1] and len(normalise(raw)) >= 2:
                best = (raw, conf)
        if preprocess and len(normalise(best[0])) < 4 and crop.shape[0] / max(crop.shape[1], 1) > 0.45:
            raw2, conf2 = self._det_rec(crop)  # square-ish plate: try two-line reading
            if len(normalise(raw2)) > len(normalise(best[0])):
                best = (raw2, conf2)
        return PlateRead(normalise(best[0]), best[1], best[0])


def _supports_params() -> bool:
    try:
        import inspect

        from rapidocr import RapidOCR

        return "params" in inspect.signature(RapidOCR.__init__).parameters
    except Exception:
        return False


def vote(reads: list[PlateRead], min_len: int = 4) -> PlateRead | None:
    """Confidence-weighted majority vote across reads of one vehicle.
    The most common length wins (weighted), then each character position is voted independently."""
    good = [r for r in reads if len(r.text) >= min_len and r.conf > 0.3]
    if not good:
        return None
    by_len: dict[int, float] = defaultdict(float)
    for r in good:
        by_len[len(r.text)] += r.conf
    length = max(by_len, key=lambda k: by_len[k])
    same = [r for r in good if len(r.text) == length]
    chars = []
    agree = []
    for i in range(length):
        c: Counter = Counter()
        for r in same:
            c[r.text[i]] += r.conf
        ch, w = c.most_common(1)[0]
        chars.append(ch)
        agree.append(w / sum(c.values()))
    return PlateRead("".join(chars), round(float(np.mean([r.conf for r in same]) * np.mean(agree)), 4), f"{len(same)} reads")
