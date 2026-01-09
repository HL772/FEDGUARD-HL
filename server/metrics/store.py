import asyncio
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket

from server.metrics.schema import normalize_round_metric


class MetricsAgent:
    def __init__(self, jsonl_path: str, clear_on_start: bool = False) -> None:
        self._metrics: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._connections: Set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._jsonl_path = Path(jsonl_path)
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if clear_on_start:
            self._jsonl_path.write_text("", encoding="utf-8")

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def record(self, metric: Dict[str, Any]) -> None:
        normalized = normalize_round_metric(metric)
        with self._lock:
            self._metrics.append(normalized)
            with self._jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(normalized) + "\n")
        self._broadcast(normalized)

    def latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._metrics[-1] if self._metrics else None

    def all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._metrics)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)
        latest = self.latest()
        if latest is not None:
            await websocket.send_json(latest)

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    def _broadcast(self, metric: Dict[str, Any]) -> None:
        if not self._loop or not self._connections:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast_async(metric), self._loop)

    async def _broadcast_async(self, metric: Dict[str, Any]) -> None:
        to_remove: List[WebSocket] = []
        for websocket in list(self._connections):
            try:
                await websocket.send_json(metric)
            except Exception:
                to_remove.append(websocket)
        for websocket in to_remove:
            self._connections.discard(websocket)
