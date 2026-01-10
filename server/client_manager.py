import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple


# ClientManagerModule（AGENT.md 3.1.B）：
# - 注册与心跳
# - 在线/可参与资格维护
# - 拉黑管理（恶意检测 + 手动管理）
class ClientManagerModule:
    def __init__(self, online_ttl_sec: float = 60.0) -> None:
        self._clients: Dict[str, Dict[str, object]] = {}
        self._blacklist: Dict[str, Dict[str, object]] = {}
        self._lock = threading.Lock()
        self._online_ttl_sec = online_ttl_sec

    def register(self, client_name: Optional[str] = None) -> Tuple[str, bool]:
        # 注册客户端；若重名则复用 client_id（便于演示与复现）
        now = time.time()
        name = client_name or ""
        with self._lock:
            if name:
                for client_id, info in self._clients.items():
                    if info.get("client_name") == name:
                        info["last_seen"] = now
                        return client_id, True
            client_id = str(uuid.uuid4())
            self._clients[client_id] = {
                "client_name": name,
                "last_seen": now,
                "score": 0.0,
                "selected_cnt": 0,
                "timeout_cnt": 0,
                "last_selected_round": -1,
            }
            return client_id, False

    def heartbeat(self, client_id: str, timestamp: Optional[float] = None) -> bool:
        # 更新 last_seen，维持在线状态
        now = timestamp if timestamp is not None else time.time()
        with self._lock:
            if client_id not in self._clients:
                return False
            self._clients[client_id]["last_seen"] = float(now)
            return True

    def is_registered(self, client_id: str) -> bool:
        with self._lock:
            return client_id in self._clients

    def get_client_name(self, client_id: str) -> str:
        with self._lock:
            return str(self._clients.get(client_id, {}).get("client_name", ""))

    def online_clients(self) -> List[Dict[str, object]]:
        # 返回在线客户端，用于 Dashboard 展示
        now = time.time()
        clients: List[Dict[str, object]] = []
        with self._lock:
            for client_id, info in self._clients.items():
                last_seen = float(info.get("last_seen", 0.0))
                if now - last_seen <= self._online_ttl_sec:
                    clients.append(
                        {
                            "client_id": client_id,
                            "client_name": info.get("client_name", ""),
                            "last_seen": last_seen,
                            "blacklisted": client_id in self._blacklist,
                        }
                    )
        return clients

    def all_clients(self) -> List[Dict[str, object]]:
        # 返回全部客户端并标注在线状态（用于管理与状态展示）
        now = time.time()
        clients: List[Dict[str, object]] = []
        with self._lock:
            for client_id, info in self._clients.items():
                last_seen = float(info.get("last_seen", 0.0))
                online = now - last_seen <= self._online_ttl_sec
                clients.append(
                    {
                        "client_id": client_id,
                        "client_name": info.get("client_name", ""),
                        "last_seen": last_seen,
                        "blacklisted": client_id in self._blacklist,
                        "online": online,
                    }
                )
        clients.sort(
            key=lambda item: str(item.get("client_name") or item.get("client_id") or "")
        )
        return clients

    def eligible_client_ids(self) -> List[str]:
        # 在线且未被拉黑的客户端（可参与训练）
        now = time.time()
        eligible: List[str] = []
        with self._lock:
            for client_id, info in self._clients.items():
                last_seen = float(info.get("last_seen", 0.0))
                if client_id in self._blacklist:
                    continue
                if now - last_seen <= self._online_ttl_sec:
                    eligible.append(client_id)
        return eligible

    def registered_client_ids(self) -> List[str]:
        with self._lock:
            return sorted(
                client_id
                for client_id in self._clients
                if client_id not in self._blacklist
            )

    def blacklist_clients(self, client_ids: List[str], reason: str = "anomaly") -> None:
        # 拉黑客户端并记录原因（不再具备参与资格）
        now = time.time()
        with self._lock:
            for client_id in client_ids:
                self._blacklist[client_id] = {"reason": reason, "since": now}

    def unblacklist_clients(self, client_ids: List[str]) -> None:
        # 手动解除拉黑
        with self._lock:
            for client_id in client_ids:
                self._blacklist.pop(client_id, None)

    def is_blacklisted(self, client_id: str) -> bool:
        with self._lock:
            return client_id in self._blacklist

    def record_selected(self, client_ids: List[str], round_id: int) -> None:
        # 记录被选中次数与最近轮次（用于采样策略）
        with self._lock:
            for client_id in client_ids:
                if client_id not in self._clients:
                    continue
                info = self._clients[client_id]
                info["selected_cnt"] = int(info.get("selected_cnt", 0)) + 1
                info["last_selected_round"] = int(round_id)

    def record_timeouts(self, client_ids: List[str]) -> None:
        # 记录超时次数（用于采样惩罚）
        with self._lock:
            for client_id in client_ids:
                if client_id not in self._clients:
                    continue
                info = self._clients[client_id]
                info["timeout_cnt"] = int(info.get("timeout_cnt", 0)) + 1

    def update_scores(
        self,
        score_updates: Dict[str, Dict[str, float]],
        ema: float = 0.2,
        timeout_penalty: float = 0.5,
        anomaly_penalty: float = 1.0,
    ) -> None:
        ema = max(0.0, min(1.0, float(ema)))
        with self._lock:
            for client_id, update in score_updates.items():
                if client_id not in self._clients:
                    continue
                info = self._clients[client_id]
                loss = float(update.get("loss", 0.0))
                acc = float(update.get("accuracy", 0.0))
                timely = float(update.get("timely", 1.0))
                excluded = float(update.get("excluded", 0.0))
                utility = acc - loss
                penalty = 0.0
                if timely <= 0.0:
                    penalty += timeout_penalty
                if excluded > 0.0:
                    penalty += anomaly_penalty
                target = utility - penalty
                prev_score = float(info.get("score", 0.0))
                info["score"] = (1.0 - ema) * prev_score + ema * target

    def sampling_state(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            state: Dict[str, Dict[str, float]] = {}
            for client_id, info in self._clients.items():
                state[client_id] = {
                    "score": float(info.get("score", 0.0)),
                    "selected_cnt": float(info.get("selected_cnt", 0)),
                    "timeout_cnt": float(info.get("timeout_cnt", 0)),
                    "last_selected_round": float(info.get("last_selected_round", -1)),
                }
            return state

    def stats(self) -> Dict[str, int]:
        with self._lock:
            registered = len(self._clients)
            blacklisted = len(self._blacklist)
        online = len(self.online_clients())
        eligible = len(self.eligible_client_ids())
        return {
            "registered": registered,
            "online": online,
            "eligible": eligible,
            "blacklisted": blacklisted,
        }
