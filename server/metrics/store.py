import asyncio
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket

from server.metrics.schema import normalize_round_metric

# MetricsAgent（AGENT.md 3.1.F）：
# - 轮次指标 JSONL 持久化
# - WebSocket 实时推送


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
        # 绑定事件循环用于 WS 广播调度
        self._loop = loop

    def record(self, metric: Dict[str, Any]) -> None:
        # 归一化、落盘并广播本轮指标
        normalized = normalize_round_metric(metric)
        with self._lock:
            self._metrics.append(normalized)
            with self._jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(normalized) + "\n")
        self._broadcast(normalized)

    def latest(self) -> Optional[Dict[str, Any]]:
        # 返回最新指标
        with self._lock:
            return self._metrics[-1] if self._metrics else None

    def all(self) -> List[Dict[str, Any]]:
        # 返回所有指标快照
        with self._lock:
            return list(self._metrics)

    def reset(self, clear_file: bool = True) -> None:
        # 清空内存指标（可选清空 JSONL 文件）
        with self._lock:
            self._metrics = []
            if clear_file:
                self._jsonl_path.write_text("", encoding="utf-8")

    async def connect(self, websocket: WebSocket) -> None:
        # 连接 WS 并推送最新指标
        await websocket.accept()
        self._connections.add(websocket)
        latest = self.latest()
        if latest is not None:
            await websocket.send_json(latest)

    def disconnect(self, websocket: WebSocket) -> None:
        # 移除 WS 连接
        self._connections.discard(websocket)

    def _broadcast(self, metric: Dict[str, Any]) -> None:
        # 异步广播，避免阻塞训练主流程
        if not self._loop or not self._connections:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast_async(metric), self._loop)

    async def _broadcast_async(self, metric: Dict[str, Any]) -> None:
        # 推送给所有订阅者并剔除断开连接
        to_remove: List[WebSocket] = []
        for websocket in list(self._connections):
            try:
                await websocket.send_json(metric)
            except Exception:
                to_remove.append(websocket)
        for websocket in to_remove:
            self._connections.discard(websocket)
