"""Download every public dataset / video the project uses (nothing large is committed to git).

    python data/download_data.py           # videos + plate datasets + OpenALPR benchmark
    python data/prepare_plates.py          # convert the plate dataset to YOLO format for training

Sources and licences: data/SOURCES.md
"""
from __future__ import annotations

import subprocess
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"
VIDEOS = {
    "vehicles.mp4": "https://media.roboflow.com/supervision/video-examples/vehicles.mp4",
    "vehicles-2.mp4": "https://media.roboflow.com/supervision/video-examples/vehicles-2.mp4",
}
HF = "https://huggingface.co/datasets/keremberke/license-plate-object-detection/resolve/main/data/{split}.zip"


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"have {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (smart-traffic-analytics demo)"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    print(f"downloaded {dest.name} ({dest.stat().st_size // 1_000_000} MB)")


def main() -> None:
    for name, url in VIDEOS.items():
        fetch(url, RAW / name)
    for split in ("train", "valid", "test"):
        z = RAW / "plates_hf" / f"{split}.zip"
        fetch(HF.format(split=split), z)
        out = RAW / "plates_hf" / split
        if not out.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out)
    bench = RAW / "openalpr-benchmarks"
    if not bench.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/openalpr/benchmarks", str(bench)], check=True)


if __name__ == "__main__":
    main()
