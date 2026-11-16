"""Settings (env prefix TRAFFIC_) and camera definitions (cameras.json)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(PROJECT_ROOT / ".env"), env_prefix="TRAFFIC_", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data" / "runtime"
    cameras_file: Path = PROJECT_ROOT / "cameras.json"
    vehicle_model: str = str(PROJECT_ROOT / "models" / "yolo11s.pt")  # COCO-pretrained; car / motorcycle / bus / truck
    plate_model: Path = PROJECT_ROOT / "models" / "plate_yolo11n.pt"
    device: str = "cuda:0"  # cuda:0 | cpu
    half: bool = True  # FP16 inference on GPU
    detect_conf: float = 0.3
    plate_conf: float = 0.35
    autostart: bool = True  # start all cameras when the API starts
    realtime: bool = True  # pace video files at their native FPS (False = as fast as possible, for benchmarks)
    stream_width: int = 1280  # width of the annotated MJPEG stream
    cors_origins: str = "http://localhost:3400,http://127.0.0.1:3400"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "traffic.db"

    @property
    def snapshot_dir(self) -> Path:
        return self.data_dir / "snapshots"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    s.snapshot_dir.mkdir(parents=True, exist_ok=True)
    return s


class DirectionZone(BaseModel):
    """Lane / carriageway polygon with the legal direction of travel.
    'toward' = vehicles must move toward the vanishing point (away from camera), 'away' = away from it (toward camera)."""
    name: str
    polygon: list[tuple[float, float]]
    legal: str  # toward | away


class CameraConfig(BaseModel):
    id: str
    name: str
    source: str  # video file (relative to project root) or rtsp:// URL
    loop: bool = True
    infer_width: int = 1280  # frames are resized to this width for vehicle detection
    vehicle_classes: list[int] = Field(default_factory=lambda: [2, 3, 5, 7])  # COCO car, motorcycle, bus, truck
    # speed: perspective calibration (image quad → real-world rectangle in metres), full-resolution pixel coordinates
    speed_source: list[tuple[float, float]] | None = None
    speed_target_m: tuple[float, float] | None = None  # (width, length) in metres
    speed_limit_kmh: float | None = None
    vanishing_point: tuple[float, float] | None = None
    direction_zones: list[DirectionZone] = Field(default_factory=list)
    no_parking_zones: list[list[tuple[float, float]]] = Field(default_factory=list)
    parking_seconds: float = 10.0  # stationary this long inside a no-parking zone = violation
    count_line: tuple[tuple[float, float], tuple[float, float]] | None = None
    anpr: bool = False
    anpr_min_box_h: int = 160  # only read plates on vehicles at least this tall (full-res px) - near the camera
    description: str = ""


def load_cameras() -> list[CameraConfig]:
    path = get_settings().cameras_file
    return [CameraConfig(**c) for c in json.loads(path.read_text(encoding="utf-8"))]
