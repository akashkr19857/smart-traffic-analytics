"""Build a SIMULATED incident clip from the public highway video, to test the wrong-way and illegal-parking rules.

Real footage rarely contains these incidents, so the clip is clearly synthetic:
* part 1 (8 s): a time-reversed segment of vehicles.mp4 - every moving vehicle now drives against the legal direction
* part 2 (18 s): normal traffic, with a car image cut from the same video pasted, stationary, on the northbound hard
  shoulder from t = 2 s (a "parked" vehicle)

    python data/make_scenarios.py   → data/scenarios/incidents.mp4 (+ incidents.json with the ground truth)
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "raw" / "vehicles.mp4"
OUT = HERE / "scenarios"
# a dark car on the northbound carriageway in frame 25 of vehicles.mp4 (4K px) and where it is pasted on the shoulder
CAR_FRAME, CAR_BOX = 25, (2585, 1055, 2800, 1225)
PARK_AT = (3060, 1245)  # top-left corner of the pasted car, inside the northbound no-parking zone


def ffmpeg() -> str:
    import sys

    sys.path.insert(0, str(HERE.parent / "demo"))
    from recorder import ffmpeg_path

    return ffmpeg_path()


def read_frames(cap: cv2.VideoCapture, start: int, n: int) -> list[bytes]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes())
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(SRC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, CAR_FRAME)
    _, ref = cap.read()
    x1, y1, x2, y2 = CAR_BOX
    car = ref[y1:y2, x1:x2].copy()
    mask = np.zeros(car.shape[:2], np.float32)
    cv2.ellipse(mask, (car.shape[1] // 2, car.shape[0] // 2), (car.shape[1] // 2 - 4, car.shape[0] // 2 - 4), 0, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)[..., None]
    raw = OUT / "_incidents_raw.avi"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    reversed_part = read_frames(cap, 300, int(8 * fps))[::-1]
    for b in reversed_part:
        vw.write(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR))
    n_rev = len(reversed_part)
    park_frames = read_frames(cap, 20, int(18 * fps))
    px, py = PARK_AT
    for i, b in enumerate(park_frames):
        f = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if i >= int(2 * fps):
            roi = f[py:py + car.shape[0], px:px + car.shape[1]].astype(np.float32)
            f[py:py + car.shape[0], px:px + car.shape[1]] = (mask * car + (1 - mask) * roi).astype(np.uint8)
        vw.write(f)
    vw.release()
    out = OUT / "incidents.mp4"
    subprocess.run([ffmpeg(), "-y", "-v", "error", "-i", str(raw), "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(out)],
                   check=True)
    raw.unlink()
    truth = {"fps": fps, "wrong_way_segment_s": [0, round(n_rev / fps, 2)],
             "parked_vehicle": {"from_s": round((n_rev + 2 * fps) / fps, 2), "to_s": round((n_rev + len(park_frames)) / fps, 2),
                                "box_4k": [px, py, px + car.shape[1], py + car.shape[0]]},
             "note": "SIMULATED: part 1 is time-reversed real footage; the parked car in part 2 is pasted from the same video."}
    (OUT / "incidents.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    print(json.dumps(truth, indent=1))


if __name__ == "__main__":
    main()
