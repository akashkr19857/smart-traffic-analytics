"""Convert the COCO-format license-plate dataset (CC BY 4.0) into a YOLO detection dataset.

    python data/prepare_plates.py     → data/plates_yolo/{images,labels}/{train,val,test} + data/plates_yolo/data.yaml
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "raw" / "plates_hf"
OUT = HERE / "plates_yolo"
SPLITS = {"train": "train", "valid": "val", "test": "test"}


def convert(split: str, name: str) -> tuple[int, int]:
    coco = json.loads((SRC / split / "_annotations.coco.json").read_text(encoding="utf-8"))
    imgs = {i["id"]: i for i in coco["images"]}
    boxes: dict[int, list] = {}
    for a in coco["annotations"]:
        boxes.setdefault(a["image_id"], []).append(a["bbox"])
    (OUT / "images" / name).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / name).mkdir(parents=True, exist_ok=True)
    n_boxes = 0
    for iid, im in imgs.items():
        src = SRC / split / im["file_name"]
        dst = OUT / "images" / name / im["file_name"]
        if not dst.exists():
            try:
                os.link(src, dst)  # hard link: no extra disk space
            except OSError:
                shutil.copy2(src, dst)
        w, h = im["width"], im["height"]
        lines = []
        for x, y, bw, bh in boxes.get(iid, []):
            if bw <= 1 or bh <= 1:
                continue
            lines.append(f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
        n_boxes += len(lines)
        (OUT / "labels" / name / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
    return len(imgs), n_boxes


def main() -> None:
    for split, name in SPLITS.items():
        n, b = convert(split, name)
        print(f"{name}: {n} images, {b} plates")
    (OUT / "data.yaml").write_text(f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: license_plate\n",
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
