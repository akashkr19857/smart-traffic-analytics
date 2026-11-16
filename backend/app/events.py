"""Tiny pub/sub that lets worker threads push live events (ingestion progress, new traces) to WebSocket clients."""
from __future__ import annotations

import asyncio
import threading
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._subs: set[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        with self._lock:
            self._subs.add((asyncio.get_running_loop(), q))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subs = {s for s in self._subs if s[1] is not q}

    def publish(self, event: dict[str, Any]) -> None:
        """Thread-safe; may be called from any thread."""
        with self._lock:
            subs = list(self._subs)
        for loop, q in subs:
            def _put(q: asyncio.Queue = q) -> None:
                if not q.full():
                    q.put_nowait(event)
            try:
                loop.call_soon_threadsafe(_put)
            except RuntimeError:  # loop closed
                self.unsubscribe(q)


bus = EventBus()
