"""SQLite persistence: vehicle passages, events (plate reads + violations), watchlist, benchmark runs."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import get_settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


def _row(obj: Any) -> dict[str, Any]:
    return {c.name: (v.isoformat() if isinstance(v := getattr(obj, c.name), datetime) else v) for c in obj.__table__.columns}


class Vehicle(Base):
    """One tracked vehicle passage through a camera view."""
    __tablename__ = "vehicles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    track_id: Mapped[int] = mapped_column(Integer)
    run_id: Mapped[str] = mapped_column(String(32), index=True)
    cls: Mapped[str] = mapped_column(String(16), index=True)
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    video_t: Mapped[float] = mapped_column(Float, default=0.0)  # seconds into the source video
    direction: Mapped[str] = mapped_column(String(32), default="")
    avg_speed_kmh: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_speed_kmh: Mapped[float | None] = mapped_column(Float, nullable=True)
    plate: Mapped[str] = mapped_column(String(16), default="", index=True)
    plate_conf: Mapped[float] = mapped_column(Float, default=0.0)
    plate_reads: Mapped[int] = mapped_column(Integer, default=0)
    counted: Mapped[bool] = mapped_column(Boolean, default=False)

    def to_dict(self) -> dict[str, Any]:
        return _row(self)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    type: Mapped[str] = mapped_column(String(32), index=True)  # plate | speeding | wrong_way | illegal_parking | watchlist
    severity: Mapped[str] = mapped_column(String(8), default="info")  # info | warning | critical
    track_id: Mapped[int] = mapped_column(Integer)
    cls: Mapped[str] = mapped_column(String(16), default="")
    plate: Mapped[str] = mapped_column(String(16), default="", index=True)
    plate_conf: Mapped[float] = mapped_column(Float, default=0.0)
    speed_kmh: Mapped[float | None] = mapped_column(Float, nullable=True)
    message: Mapped[str] = mapped_column(Text, default="")
    snapshot: Mapped[str] = mapped_column(Text, default="")  # file name in data/runtime/snapshots
    plate_crop: Mapped[str] = mapped_column(Text, default="")
    video_t: Mapped[float] = mapped_column(Float, default=0.0)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    def to_dict(self) -> dict[str, Any]:
        return _row(self)


class Watchlist(Base):
    __tablename__ = "watchlist"
    plate: Mapped[str] = mapped_column(String(16), primary_key=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return _row(self)


_engine = None
_Session: sessionmaker[Session] | None = None


def session() -> Session:
    global _engine, _Session
    if _engine is None:
        _engine = create_engine(f"sqlite:///{get_settings().db_path.as_posix()}", connect_args={"check_same_thread": False})

        @event.listens_for(_engine, "connect")
        def _pragma(conn, _):  # noqa: ANN001
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

        Base.metadata.create_all(_engine)
        _Session = sessionmaker(_engine, expire_on_commit=False)
    assert _Session is not None
    return _Session()


def reset_engine() -> None:
    global _engine, _Session
    if _engine is not None:
        _engine.dispose()
    _engine, _Session = None, None
