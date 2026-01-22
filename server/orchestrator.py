import copy
import math
import random
import threading
import time
from fnmatch import fnmatch
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import torch

from server.aggregation.fedavg import (
    AggregationModule,
    apply_delta,
    apply_delta_partial,
    create_model,
    init_model_state,
)
from server.aggregation.secure_agg import SecureAggregationModule
from server.privacy.accountant import RDPAccountant
from server.security.malicious_detect import MaliciousDetectionModule


# CoordinatorModule：
# - 轮次调度、客户端采样、模型下发、聚合触发
# - DP 参数下发、超时处理、指标记录
class CoordinatorModule:
    def __init__(
        self,
        max_rounds: int,
        clients_per_round: int,
        lr: float = 0.1,
        batch_size: int = 32,
        epochs: int = 1,
        secure_aggregation: bool = False,
        malicious_detection: bool = True,
        robust_agg_method: str = "fedavg",
        trim_ratio: float = 0.2,
        loss_threshold: float = 3.0,
        norm_threshold: float = 3.0,
        require_both: bool = True,
        min_mad: float = 0.05,
        cosine_threshold: float = 2.5,
        cosine_enabled: bool = False,
        cosine_top_k: int = 5,
        metrics_agent: Optional[object] = None,
        dp_config: Optional[Dict[str, object]] = None,
        compression_config: Optional[Dict[str, object]] = None,
        client_provider: Optional[Callable[[], List[Dict[str, object]]]] = None,
        client_name_lookup: Optional[Callable[[str], str]] = None,
        eligible_clients_provider: Optional[Callable[[], List[str]]] = None,
        exclude_callback: Optional[Callable[[List[str]], None]] = None,
        client_stats_provider: Optional[Callable[[], Dict[str, int]]] = None,
        fedprox_mu: float = 0.0,
        deadline_ms: float = 0.0,
        train_algo: str = "fedavg",
        model_split: Optional[Dict[str, object]] = None,
        public_head_eval: Optional[Dict[str, object]] = None,
        sampling_config: Optional[Dict[str, object]] = None,
        sampling_state_provider: Optional[Callable[[], Dict[str, Dict[str, float]]]] = None,
        sampling_record_selected: Optional[Callable[[List[str], int], None]] = None,
        sampling_record_timeout: Optional[Callable[[List[str]], None]] = None,
        sampling_update_scores: Optional[Callable[[Dict[str, Dict[str, float]]], None]] = None,
        byzantine_f: int = 1,
        server_lr: float = 1.0,
        server_update_clip: float = 0.0,
        timeout_simulation: Optional[Dict[str, object]] = None,
    ) -> None:
        self.max_rounds = max_rounds
        self.clients_per_round = clients_per_round
        self.round_config = {
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "secure_aggregation": secure_aggregation,
            "mu": fedprox_mu,
            "deadline_ms": deadline_ms,
            "algo": str(train_algo).lower(),
        }
        self.current_round = 0
        self.model_state = init_model_state()
        self.clients: Set[str] = set()
        self.active_clients: List[str] = []
        self.received_clients: Set[str] = set()
        self.pending_updates: List[Dict[str, object]] = []
        self.training_done = False
        self.round_start_time: Optional[float] = None
        self.round_deadline: Optional[float] = None
        self._deadline_timer: Optional[threading.Timer] = None
        self._eval_loader = None
        self._lock = threading.Lock()
        # AggregationModule：FedAvg + 鲁棒聚合
        self._aggregator = AggregationModule()
        self._secure_agg = SecureAggregationModule()
        self._forced_timeouts: Set[str] = set()
        self._cooldown_until: Dict[str, int] = {}
        self._cosine_reference: Optional[Dict[str, list]] = None
        # MaliciousDetectionModule：统计异常 + 余弦相似度检测
        self._detector = MaliciousDetectionModule(
            loss_threshold=loss_threshold,
            norm_threshold=norm_threshold,
            require_both=require_both,
            min_mad=min_mad,
            cosine_threshold=cosine_threshold,
            cosine_enabled=cosine_enabled,
            cosine_top_k=cosine_top_k,
        )
        self.secure_aggregation = secure_aggregation
        self.malicious_detection = malicious_detection
        self.robust_agg_method = robust_agg_method
        self.trim_ratio = trim_ratio
        self.robust_agg_byzantine = int(byzantine_f)
        self.metrics_agent = metrics_agent
        self.dp_config = dp_config or {}
        self.compression_config = compression_config or {}
        self.client_provider = client_provider
        self.client_name_lookup = client_name_lookup
        self.eligible_clients_provider = eligible_clients_provider
        self.exclude_callback = exclude_callback
        self.client_stats_provider = client_stats_provider
        self.deadline_ms = float(deadline_ms)
        self.train_algo = str(train_algo).lower()
        self.model_split = model_split or {}
        self.timeout_simulation = timeout_simulation or {}
        self.timeout_cooldown_rounds = int(
            self.timeout_simulation.get("cooldown_rounds", 0)
        )
        if self.train_algo == "fedper" and not self.model_split:
            self.model_split = {"backbone_keys": ["backbone.*"], "head_keys": ["head.*"]}
        self._backbone_keys, self._head_keys = self._resolve_split_keys(
            self.model_state.keys(), self.model_split
        )
        private_patterns = self._normalize_pattern_list(self.model_split.get("private_head_keys"))
        if self.train_algo == "fedper_dual" and not private_patterns:
            private_patterns = ["private_head.*"]
        self._private_head_keys = self._match_patterns(self.model_state.keys(), private_patterns)
        self.public_head_eval = public_head_eval or {}
        self._public_loader = None
        self.sampling_config = sampling_config or {}
        self.sampling_enabled = bool(self.sampling_config.get("enabled", False))
        self.sampling_strategy = str(self.sampling_config.get("strategy", "random")).lower()
        self.sampling_selection_mode = "pre"
        self.sampling_epsilon = float(self.sampling_config.get("epsilon", 0.1))
        self.sampling_score_ema = float(self.sampling_config.get("score_ema", 0.2))
        self.sampling_timeout_penalty = float(
            self.sampling_config.get("timeout_penalty", 0.5)
        )
        self.sampling_anomaly_penalty = float(
            self.sampling_config.get("anomaly_penalty", 1.0)
        )
        self.sampling_fairness_window = int(
            self.sampling_config.get("fairness_window", 3)
        )
        self.sampling_softmax_temp = float(
            self.sampling_config.get("softmax_temp", 1.0)
        )
        self.sampling_state_provider = sampling_state_provider
        self.sampling_record_selected = sampling_record_selected
        self.sampling_record_timeout = sampling_record_timeout
        self.sampling_update_scores = sampling_update_scores
        self._round_selection_mode = "pre"
        self.dp_enabled = bool(self.dp_config.get("enabled", False))
        self.dp_mode = str(self.dp_config.get("mode", "static"))
        self.dp_delta = float(self.dp_config.get("delta", 1e-5))
        self.dp_noise_multiplier = float(self.dp_config.get("noise_multiplier", 0.0))
        self.dp_clip_norm = float(self.dp_config.get("clip_norm", 1.0))
        self.dp_target_epsilon = float(self.dp_config.get("target_epsilon", 0.0))
        self.dp_schedule = self.dp_config.get("schedule", {})
        self.dp_adaptive_clip = self.dp_config.get("adaptive_clip", {})
        self._clip_norm_ema = self.dp_clip_norm
        self._current_dp_params: Dict[str, object] = {}
        self._last_epsilon_accountant = 0.0
        self._accountant = (
            RDPAccountant()
            if self.dp_enabled and self.dp_mode == "adaptive_rdp"
            else None
        )
        self.server_lr = float(server_lr)
        self.server_update_clip = float(server_update_clip)

    def register_client(self, client_id: str) -> None:
        # 记录注册客户端（用于参与资格判断）
        with self._lock:
            self.clients.add(client_id)

    def _resolve_split_keys(
        self, keys: Iterable[str], model_split: Dict[str, object]
    ) -> Tuple[Set[str], Set[str]]:
        key_list = list(keys)
        backbone_patterns = self._normalize_pattern_list(model_split.get("backbone_keys"))
        head_patterns = self._normalize_pattern_list(model_split.get("head_keys"))

        if not backbone_patterns and not head_patterns:
            return set(key_list), set()

        backbone = self._match_patterns(key_list, backbone_patterns)
        head = self._match_patterns(key_list, head_patterns)
        if backbone and not head_patterns:
            head = set(key_list) - backbone
        if head and not backbone_patterns:
            backbone = set(key_list) - head
        if not backbone:
            backbone = set(key_list)
        return backbone, head

    @staticmethod
    def _normalize_pattern_list(value: object) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]

    @staticmethod
    def _match_patterns(keys: Iterable[str], patterns: List[str]) -> Set[str]:
        if not patterns:
            return set()
        matched: Set[str] = set()
        for key in keys:
            for pattern in patterns:
                if fnmatch(key, pattern):
                    matched.add(key)
                    break
        return matched

    def _filter_state(self, state: Dict[str, list], keys: Set[str]) -> Dict[str, list]:
        if not keys:
            return state
        return {key: value for key, value in state.items() if key in keys}

    def _aggregated_keys(self) -> Set[str]:
        if self.train_algo == "fedper_dual":
            return set(self._backbone_keys) | set(self._head_keys)
        if self.train_algo == "fedper":
            return set(self._backbone_keys)
        return set()

    def _eligible_clients(self) -> List[str]:
        # 可参与客户端：在线且未拉黑（ClientManagerModule）
        if self.eligible_clients_provider is not None:
            eligible = list(self.eligible_clients_provider())
        else:
            eligible = list(self.clients)
        if not eligible:
            return []
        self._cleanup_cooldowns()
        if self._cooldown_until:
            eligible = [
                client_id
                for client_id in eligible
                if not self._is_in_cooldown(client_id)
            ]
        return sorted(eligible)

    def _cleanup_cooldowns(self) -> None:
        if not self._cooldown_until:
            return
        expired = [
            client_id
            for client_id, until_round in self._cooldown_until.items()
            if self.current_round > int(until_round)
        ]
        for client_id in expired:
            self._cooldown_until.pop(client_id, None)

    def _is_in_cooldown(self, client_id: str, round_id: Optional[int] = None) -> bool:
        if not self._cooldown_until:
            return False
        round_id = self.current_round if round_id is None else int(round_id)
        until_round = self._cooldown_until.get(client_id)
        return until_round is not None and round_id <= int(until_round)

    def _apply_timeout_cooldown(self, client_ids: List[str]) -> None:
        cooldown_rounds = int(self.timeout_cooldown_rounds)
        if cooldown_rounds <= 0 or not client_ids:
            return
        until_round = self.current_round + cooldown_rounds
        for client_id in client_ids:
            prev_until = int(self._cooldown_until.get(client_id, 0))
            if until_round > prev_until:
                self._cooldown_until[client_id] = until_round

    def _forced_timeout_clients(self, active_clients: List[str]) -> Set[str]:
        # 超时演示：指定客户端在本轮模拟超时
        cfg = self.timeout_simulation or {}
        if not bool(cfg.get("enabled", False)):
            return set()
        rounds = cfg.get("rounds", [])
        if rounds and self.current_round not in {int(r) for r in rounds}:
            return set()
        ids = {str(cid) for cid in cfg.get("client_ids", []) if cid}
        rank_values = cfg.get("client_ranks", None)
        if rank_values is None:
            rank_values = cfg.get("malicious_ranks", [])
        ranks = {int(rank) for rank in rank_values or []}
        forced: Set[str] = set()
        for client_id in active_clients:
            if client_id in ids:
                forced.add(client_id)
                continue
            if ranks and self.client_name_lookup:
                name = self.client_name_lookup(client_id)
                if name.startswith("client-"):
                    try:
                        rank = int(name.split("-", 1)[1]) - 1
                    except ValueError:
                        continue
                    if rank in ranks:
                        forced.add(client_id)
        return forced

    def _expected_client_count(self) -> int:
        return max(len(self.active_clients) - len(self._forced_timeouts), 0)

    def _select_clients(self, eligible_clients: List[str]) -> List[str]:
        # 客户端采样（通信优化要求）
        if not eligible_clients:
            return []
        if not self.sampling_enabled or self.sampling_strategy == "random":
            if len(eligible_clients) <= self.clients_per_round:
                return list(eligible_clients)
            return random.sample(list(eligible_clients), self.clients_per_round)

        state = self.sampling_state_provider() if self.sampling_state_provider else {}  # 拉取历史采样状态
        scores = {
            client_id: float(state.get(client_id, {}).get("score", 0.0))
            for client_id in eligible_clients
        }
        last_selected = {
            client_id: int(state.get(client_id, {}).get("last_selected_round", -1))
            for client_id in eligible_clients
        }
        selected: List[str] = []
        remaining = list(eligible_clients)
        if self.sampling_fairness_window > 0:  # 公平窗口：保证长时间未选中的客户端被补选
            overdue = [
                client_id
                for client_id in eligible_clients
                if last_selected.get(client_id, -1) < 0
                or self.current_round - last_selected.get(client_id, -1)
                >= self.sampling_fairness_window
            ]
            if overdue:
                if len(overdue) >= self.clients_per_round:
                    remaining = list(overdue)  # 若超额，直接从过期集合内抽样
                else:
                    selected.extend(overdue)  # 先补齐过期客户端
                    remaining = [cid for cid in eligible_clients if cid not in selected]
        while remaining and len(selected) < self.clients_per_round:
            if random.random() < self.sampling_epsilon:
                choice = random.choice(remaining)  # 以 ε 概率随机探索
            else:
                temp = max(self.sampling_softmax_temp, 1e-6)
                max_score = max(scores.get(cid, 0.0) for cid in remaining)
                weights = [
                    math.exp((scores.get(cid, 0.0) - max_score) / temp)
                    for cid in remaining
                ]
                total = sum(weights)
                if total <= 0:
                    choice = random.choice(remaining)
                else:
                    probs = [w / total for w in weights]
                    choice = random.choices(remaining, weights=probs, k=1)[0]  # 否则按评分采样
            selected.append(choice)
            remaining.remove(choice)
        return selected

    def _schedule_noise_multiplier(self) -> float:
        # DP 噪声调度（自适应 RDP + 预设衰减）
        base = self.dp_noise_multiplier
        if (
            self.dp_mode == "adaptive_rdp"
            and self.dp_target_epsilon > 0
            and self._last_epsilon_accountant > 0
        ):
            overshoot = self._last_epsilon_accountant / self.dp_target_epsilon
            if overshoot > 1.0:
                base *= overshoot
        schedule_type = str(self.dp_schedule.get("type", "")).strip().lower()
        if not schedule_type:
            return base
        ratio = float(self.dp_schedule.get("sigma_end_ratio", 1.0))
        if self.max_rounds <= 1:
            progress = 1.0
        else:
            progress = (self.current_round - 1) / max(self.max_rounds - 1, 1)
        if schedule_type == "linear":
            factor = 1.0 - progress * (1.0 - ratio)
        elif schedule_type == "exp":
            factor = ratio ** progress
        elif schedule_type == "cosine_decay":
            factor = ratio + 0.5 * (1.0 - ratio) * (1.0 + math.cos(math.pi * progress))
        else:
            factor = 1.0
        return base * factor

    def _get_dp_params(self) -> Dict[str, object]:
        clip_norm = self._clip_norm_ema if self._clip_norm_ema is not None else self.dp_clip_norm
        params: Dict[str, object] = {
            "enabled": self.dp_enabled,
            "mode": self.dp_mode if self.dp_enabled else "off",
            "clip_norm": clip_norm,
            "noise_multiplier": self._schedule_noise_multiplier(),
            "delta": self.dp_delta,
            "target_epsilon": self.dp_target_epsilon,
        }
        return params

    def _update_clip_norm(self, update_norms: List[float]) -> float:
        # 自适应裁剪：根据客户端更新范数分布动态调整 clip_norm
        # - update_norms：本轮客户端更新的 L2 范数（来自 pre_dp_norm / update_norm）
        # - percentile：使用分位数作为目标阈值（默认 0.9）
        # - ema：用指数滑动平均平滑阈值变化
        if not update_norms:
            return float(self._clip_norm_ema or self.dp_clip_norm)
        if not bool(self.dp_adaptive_clip.get("enabled", False)):
            return float(self._clip_norm_ema or self.dp_clip_norm)
        percentile = float(self.dp_adaptive_clip.get("percentile", 0.9))
        ema = float(self.dp_adaptive_clip.get("ema", 0.2))
        percentile = max(0.0, min(1.0, percentile))
        ema = max(0.0, min(1.0, ema))
        sorted_vals = sorted(update_norms)
        # 选取分位数位置作为目标阈值
        idx = int(round(percentile * (len(sorted_vals) - 1)))
        target = sorted_vals[idx]
        if self._clip_norm_ema is None:
            # 首轮直接采用分位数阈值
            self._clip_norm_ema = target
        else:
            # EMA 平滑：防止 clip_norm 波动过大
            self._clip_norm_ema = (1.0 - ema) * self._clip_norm_ema + ema * target
        return float(self._clip_norm_ema)
    def get_round_payload(self, client_id: str) -> Dict[str, object]:
        # 下发给客户端：模型参数 + 轮次配置 + 参与列表
        with self._lock:
            if self.training_done:
                return {"status": "done"}

            if self.current_round == 0:
                eligible = self._eligible_clients()
                if len(eligible) < self.clients_per_round:
                    return {"status": "wait", "reason": "waiting_for_clients"}
                self._start_round(eligible)

            if client_id in self._forced_timeouts:
                return {"status": "wait", "reason": "timeout"}
            if client_id not in self.active_clients:
                return {"status": "wait", "reason": "not_selected"}
            if client_id in self.received_clients:
                return {"status": "wait", "reason": "update_received"}

            return {
                "status": "ready",
                "round_id": self.current_round,
                "model_state": self.model_state,
                "config": self.round_config,
                "participants": list(self.active_clients),
            }

    def submit_update(
        self,
        client_id: str,
        round_id: int,
        model_state: Optional[Dict[str, list]],
        delta_state: Optional[Dict[str, list]],
        masked_update: Optional[Dict[str, list]],
        num_samples: int,
        train_loss: float,
        update_norm: Optional[float],
        train_accuracy: Optional[float],
        epsilon: Optional[float],
        cosine_score: Optional[float],
        upload_bytes: Optional[int],
        download_bytes: Optional[int],
        raw_bytes: Optional[int],
        compressed_bytes: Optional[int],
        train_time_ms: Optional[float],
        label_histogram: Optional[Dict[str, int]] = None,
        train_steps: Optional[int] = None,
        pre_dp_norm: Optional[float] = None,
        clip_applied: Optional[bool] = None,
        attack_type: Optional[str] = None,
    ) -> Dict[str, object]:
        # 接收客户端更新并在条件满足时触发聚合
        with self._lock:
            if self.training_done:
                return {"status": "done"}
            if round_id != self.current_round:
                return {"status": "stale", "current_round": self.current_round}
            if client_id in self._forced_timeouts:
                return {"status": "rejected", "reason": "timeout"}
            if client_id not in self.active_clients:
                return {"status": "rejected", "reason": "not_selected"}
            if client_id in self.received_clients:
                return {"status": "duplicate"}

            self.received_clients.add(client_id)
            update_payload: Dict[str, object] = {
                "client_id": client_id,
                "num_samples": num_samples,
                "train_loss": train_loss,
                "update_norm": update_norm,
                "train_accuracy": train_accuracy,
                "epsilon": epsilon,
                "cosine_score": cosine_score,
                "upload_bytes": upload_bytes,
                "download_bytes": download_bytes,
                "raw_bytes": raw_bytes,
                "compressed_bytes": compressed_bytes,
                "train_time_ms": train_time_ms,
                "label_histogram": label_histogram or {},
                "train_steps": train_steps,
                "pre_dp_norm": pre_dp_norm,
                "clip_applied": clip_applied,
                "attack_type": attack_type,
            }
            if masked_update is not None:
                update_payload["masked_update"] = masked_update
            elif delta_state is not None:
                update_payload["delta_state"] = delta_state
            else:
                update_payload["model_state"] = model_state
            self.pending_updates.append(update_payload)

            if len(self.received_clients) >= self._expected_client_count():
                self._finish_round()
        return {"status": "accepted"}

    def _start_round(self, eligible_clients: Optional[List[str]] = None) -> None:
        # 启动新一轮：采样客户端并发布 DP/训练参数
        self.current_round += 1
        if eligible_clients is None:
            eligible_clients = self._eligible_clients()
        # 固定使用 pre 选择模式（已移除 post）
        self._round_selection_mode = "pre"
        self.active_clients = self._select_clients(sorted(eligible_clients))
        if self.sampling_record_selected is not None:
            self.sampling_record_selected(self.active_clients, self.current_round)
        self._forced_timeouts = self._forced_timeout_clients(self.active_clients)
        if self._forced_timeouts:
            print(
                f"[server] round {self.current_round} forced_timeouts={sorted(self._forced_timeouts)}",
                flush=True,
            )
        self.received_clients = set()
        self.pending_updates = []
        self.round_start_time = time.time()
        self.round_deadline = (
            self.round_start_time + self.deadline_ms / 1000.0
            if self.deadline_ms > 0
            else None
        )
        self._current_dp_params = self._get_dp_params()
        self.round_config["dp_params"] = self._current_dp_params
        if self.malicious_detection and self._detector.cosine_enabled and self._cosine_reference:
            self.round_config["cosine_ref"] = self._cosine_reference
        else:
            self.round_config.pop("cosine_ref", None)
        self._schedule_deadline()
        print(
            f"[server] round {self.current_round} started clients={self.active_clients}",
            flush=True,
        )

    def _schedule_deadline(self) -> None:
        if self.deadline_ms <= 0:
            return
        if self._deadline_timer is not None:
            self._deadline_timer.cancel()
        self._deadline_timer = threading.Timer(self.deadline_ms / 1000.0, self._on_deadline)
        self._deadline_timer.daemon = True
        self._deadline_timer.start()

    def _on_deadline(self) -> None:
        with self._lock:
            if self.training_done or self.current_round == 0:
                return
            if len(self.received_clients) >= self._expected_client_count():
                return
            if not self.pending_updates:
                self._schedule_deadline()
                return
            print(
                f"[server] round {self.current_round} deadline reached, finishing early",
                flush=True,
            )
            self._finish_round(deadline_triggered=True)

    def _finish_round(self, deadline_triggered: bool = False) -> None:
        # 聚合更新、异常检测、服务端评估、记录指标
        if self._deadline_timer is not None:
            self._deadline_timer.cancel()
            self._deadline_timer = None
        prev_state = copy.deepcopy(self.model_state)
        dropped_clients = [
            client_id
            for client_id in self.active_clients
            if client_id not in self.received_clients
        ]
        if dropped_clients:
            print(
                f"[server] round {self.current_round} dropped clients={dropped_clients}",
                flush=True,
            )
            if self.sampling_record_timeout is not None:
                self.sampling_record_timeout(dropped_clients)
            self._apply_timeout_cooldown(dropped_clients)
        excluded: List[str] = []
        detection: Dict[str, object] = {}
        if self.malicious_detection:
            detection = self._detector.detect(self.pending_updates)
            excluded = detection.get("excluded_clients", [])
            if excluded:
                print(
                    f"[server] round {self.current_round} excluded={excluded}",
                    flush=True,
                )

        excluded_for_agg = excluded
        if self.secure_aggregation and excluded:
            print(
                "[server] secure aggregation enabled, exclusion not applied to preserve mask cancellation",
                flush=True,
            )
            excluded_for_agg = []

        if excluded and self.exclude_callback:
            self.exclude_callback(excluded)

        excluded_set = set(excluded)
        updates_for_metrics = [
            u
            for u in self.pending_updates
            if str(u.get("client_id", "")) not in excluded_set
        ]
        if not updates_for_metrics:
            updates_for_metrics = list(self.pending_updates)

        updates = [
            u for u in self.pending_updates if u.get("client_id") not in excluded_for_agg
        ]
        if not updates:
            updates = self.pending_updates
        selected_clients = [str(update.get("client_id", "")) for update in updates]
        use_masked = any("masked_update" in update for update in updates)
        dp_mode = str(self._current_dp_params.get("mode", self.dp_mode))
        agg_total_samples = sum(update["num_samples"] for update in updates)
        loss_weight_total = sum(update["num_samples"] for update in updates_for_metrics)
        if loss_weight_total <= 0:
            loss_weight_total = max(len(updates_for_metrics), 1)
            agg_weighted_loss = sum(
                float(update["train_loss"]) for update in updates_for_metrics
            )
        else:
            agg_weighted_loss = sum(
                float(update["train_loss"]) * int(update["num_samples"])
                for update in updates_for_metrics
            )
        round_loss = agg_weighted_loss / loss_weight_total
        round_accuracy = 0.0
        if loss_weight_total > 0:
            accuracy_weighted = sum(
                float(update.get("train_accuracy", 0.0)) * int(update["num_samples"])
                for update in updates_for_metrics
            )
            round_accuracy = accuracy_weighted / loss_weight_total

        if self.sampling_update_scores is not None:
            score_updates: Dict[str, Dict[str, float]] = {}
            excluded_set = set(excluded)
            dropped_set = set(dropped_clients)
            for update in self.pending_updates:
                client_id = str(update.get("client_id", ""))
                score_updates[client_id] = {
                    "loss": float(update.get("train_loss", 0.0)),
                    "accuracy": float(update.get("train_accuracy", 0.0)),
                    "timely": 0.0 if client_id in dropped_set else 1.0,
                    "excluded": 1.0 if client_id in excluded_set else 0.0,
                }
            for client_id in dropped_clients:
                if client_id in score_updates:
                    continue
                score_updates[client_id] = {
                    "loss": 0.0,
                    "accuracy": 0.0,
                    "timely": 0.0,
                    "excluded": 0.0,
                }
            self.sampling_update_scores(
                score_updates,
                ema=self.sampling_score_ema,
                timeout_penalty=self.sampling_timeout_penalty,
                anomaly_penalty=self.sampling_anomaly_penalty,
            )

        use_delta = any("delta_state" in update for update in updates)
        update_delta: Optional[Dict[str, list]] = None
        if use_masked:
            # 安全聚合：先求和掩码更新，再按样本数归一化
            masked_updates = [update["masked_update"] for update in updates]
            summed = self._secure_agg.aggregate(masked_updates)
            averaged_delta = {
                k: (torch.tensor(v, dtype=torch.float32) / agg_total_samples).cpu().tolist()
                for k, v in summed.items()
            }
            agg_keys = self._aggregated_keys()
            if agg_keys:
                averaged_delta = self._filter_state(averaged_delta, agg_keys)
            averaged_delta = self._scale_update(averaged_delta)
            if agg_keys:
                self.model_state = apply_delta_partial(self.model_state, averaged_delta)
            else:
                self.model_state = apply_delta(self.model_state, averaged_delta)
            update_delta = averaged_delta
        else:
            # 普通/鲁棒聚合（FedAvg/trimmed/median/krum）
            method = self.robust_agg_method if not self.secure_aggregation else "fedavg"
            aggregated = self._aggregator.aggregate(
                updates,
                use_delta=use_delta,
                method=method,
                trim_ratio=self.trim_ratio,
                byzantine_f=self.robust_agg_byzantine,
            )
            if use_delta:
                agg_keys = self._aggregated_keys()
                if agg_keys:
                    aggregated = self._filter_state(aggregated, agg_keys)
                aggregated = self._scale_update(aggregated)
                if agg_keys:
                    self.model_state = apply_delta_partial(self.model_state, aggregated)
                else:
                    self.model_state = apply_delta(self.model_state, aggregated)
                update_delta = aggregated
            else:
                agg_keys = self._aggregated_keys()
                if agg_keys:
                    filtered_state = self._filter_state(aggregated, agg_keys)
                    update_delta = self._state_delta(prev_state, filtered_state)
                    update_delta = self._scale_update(update_delta)
                    self.model_state = apply_delta_partial(prev_state, update_delta)
                else:
                    update_delta = self._state_delta(prev_state, aggregated)
                    update_delta = self._scale_update(update_delta)
                    self.model_state = apply_delta(prev_state, update_delta)
        print(
            f"[server] round {self.current_round} aggregated loss={round_loss:.4f}",
            flush=True,
        )
        update_norm = self._state_norm(update_delta) if update_delta else 0.0
        if update_delta:
            self._cosine_reference = copy.deepcopy(update_delta)
        eval_loss, eval_accuracy = self._evaluate_model(self.model_state)
        update_norms = [
            float(update.get("pre_dp_norm", 0.0))
            for update in updates_for_metrics
            if update.get("pre_dp_norm") is not None
        ]
        if not update_norms:
            update_norms = [
                float(update.get("update_norm", 0.0))
                for update in updates_for_metrics
                if update.get("update_norm") is not None
            ]
        clip_norm = float(self._current_dp_params.get("clip_norm") or self.dp_clip_norm)
        if update_norms:
            clip_rate = sum(1 for value in update_norms if value > clip_norm) / len(
                update_norms
            )
        else:
            clip_rate = 0.0
        self._update_clip_norm(update_norms)
        epsilon_accountant = 0.0
        if self._accountant is not None and self.dp_enabled:
            batch_size = float(self.round_config.get("batch_size", 1))
            sample_rates = []
            step_candidates = []
            for update in updates:
                num_samples = float(update.get("num_samples") or 0.0)
                if num_samples > 0:
                    sample_rates.append(min(1.0, batch_size / num_samples))
                steps = update.get("train_steps")
                if steps is not None:
                    step_candidates.append(int(steps))
            if not step_candidates:
                epochs = int(self.round_config.get("epochs", 1))
                for update in updates:
                    num_samples = float(update.get("num_samples") or 0.0)
                    if batch_size > 0:
                        step_candidates.append(int(math.ceil(num_samples / batch_size)) * epochs)
            sample_rate = sum(sample_rates) / len(sample_rates) if sample_rates else 0.0
            steps = max(step_candidates) if step_candidates else 0
            noise_multiplier = float(self._current_dp_params.get("noise_multiplier") or 0.0)
            self._accountant.update(sample_rate, noise_multiplier, steps=steps)
            epsilon_accountant = self._accountant.get_epsilon(self.dp_delta)
            self._last_epsilon_accountant = epsilon_accountant
        self._record_metrics(
            updates=updates,
            all_updates=self.pending_updates,
            excluded=excluded,
            dropped_clients=dropped_clients,
            round_loss=round_loss,
            round_accuracy=round_accuracy,
            update_norm=update_norm,
            eval_loss=eval_loss,
            eval_accuracy=eval_accuracy,
            deadline_triggered=deadline_triggered,
            dp_mode=dp_mode,
            clip_rate=clip_rate,
            epsilon_accountant=epsilon_accountant,
            participants=selected_clients,
            detection=detection,
        )

        if self.current_round >= self.max_rounds:
            self.training_done = True
            print("[server] training complete", flush=True)
            return

        self._start_round()

    def _state_delta(self, base: Dict[str, list], new: Dict[str, list]) -> Dict[str, list]:
        delta: Dict[str, list] = {}
        for key, value in new.items():
            base_tensor = torch.tensor(base[key], dtype=torch.float32)
            new_tensor = torch.tensor(value, dtype=torch.float32)
            delta[key] = (new_tensor - base_tensor).cpu().tolist()
        return delta

    def _state_norm(self, state: Dict[str, list]) -> float:
        total = 0.0
        for value in state.values():
            tensor = torch.tensor(value, dtype=torch.float32)
            total += float(torch.sum(tensor ** 2).item())
        return math.sqrt(total)

    def _scale_update(self, delta_state: Dict[str, list]) -> Dict[str, list]:
        if not delta_state:
            return delta_state
        scale = 1.0
        if self.server_update_clip > 0:
            norm = self._state_norm(delta_state)
            if norm > self.server_update_clip:
                scale *= self.server_update_clip / max(norm, 1e-12)
        if self.server_lr > 0 and self.server_lr != 1.0:
            scale *= self.server_lr
        if scale == 1.0:
            return delta_state
        scaled: Dict[str, list] = {}
        for key, value in delta_state.items():
            tensor = torch.tensor(value, dtype=torch.float32)
            scaled[key] = (tensor * scale).cpu().tolist()
        return scaled

    def _get_eval_loader(self):
        if self._eval_loader is not None:
            return self._eval_loader
        try:
            from torch.utils.data import DataLoader
            from torchvision import datasets, transforms

            transform = transforms.ToTensor()
            dataset = datasets.MNIST(
                root="data", train=False, download=False, transform=transform
            )
            self._eval_loader = DataLoader(dataset, batch_size=256, shuffle=False)
        except Exception as exc:
            print(f"[server] eval loader unavailable: {exc}", flush=True)
            self._eval_loader = None
        return self._eval_loader

    def _get_public_loader(self):
        if self._public_loader is not None:
            return self._public_loader
        if not bool(self.public_head_eval.get("enabled", False)):
            return None
        try:
            from torch.utils.data import DataLoader
            from torchvision import datasets, transforms

            transform = transforms.ToTensor()
            data_split = str(self.public_head_eval.get("data", "train")).lower()
            use_train = data_split != "test"
            dataset = datasets.MNIST(
                root="data", train=use_train, download=False, transform=transform
            )
            batch_size = int(self.public_head_eval.get("batch_size", 128))
            self._public_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        except Exception as exc:
            print(f"[server] public loader unavailable: {exc}", flush=True)
            self._public_loader = None
        return self._public_loader

    def _train_public_head(self, model) -> None:
        if self.train_algo != "fedper":
            return
        if not bool(self.public_head_eval.get("enabled", False)):
            return
        loader = self._get_public_loader()
        if loader is None:
            return
        for param in model.backbone.parameters():
            param.requires_grad = False
        lr = float(self.public_head_eval.get("lr", 0.1))
        epochs = int(self.public_head_eval.get("epochs", 1))
        max_batches = int(self.public_head_eval.get("max_batches", 0))
        optimizer = torch.optim.SGD(model.head.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        for _ in range(max(1, epochs)):
            for batch_idx, (batch_x, batch_y) in enumerate(loader):
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                if max_batches > 0 and batch_idx + 1 >= max_batches:
                    break

    def _evaluate_model(self, model_state: Dict[str, list]) -> tuple[float, float]:
        loader = self._get_eval_loader()
        if loader is None:
            return 0.0, 0.0
        model = create_model()
        tensor_state = {k: torch.tensor(v, dtype=torch.float32) for k, v in model_state.items()}
        model.load_state_dict(tensor_state)
        self._train_public_head(model)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                total_loss += float(loss.item()) * batch_x.size(0)
                total_samples += int(batch_x.size(0))
                preds = torch.argmax(logits, dim=1)
                correct += int((preds == batch_y).sum().item())
        avg_loss = total_loss / max(total_samples, 1)
        accuracy = correct / max(total_samples, 1)
        return avg_loss, accuracy

    def _record_metrics(
        self,
        updates: List[Dict[str, object]],
        all_updates: List[Dict[str, object]],
        excluded: List[str],
        dropped_clients: List[str],
        round_loss: float,
        round_accuracy: float,
        update_norm: float,
        eval_loss: float,
        eval_accuracy: float,
        deadline_triggered: bool,
        dp_mode: str,
        clip_rate: float,
        epsilon_accountant: float,
        participants: Optional[List[str]] = None,
        detection: Optional[Dict[str, object]] = None,
    ) -> None:
        if self.metrics_agent is None:
            return
        epsilons = [float(update.get("epsilon") or 0.0) for update in updates]
        epsilon_max = max(epsilons) if epsilons else 0.0
        epsilon_avg = sum(epsilons) / len(epsilons) if epsilons else 0.0
        if participants:
            participants_list = list(participants)
        else:
            participants_list = [
                str(update.get("client_id", "")) for update in updates if update.get("client_id")
            ]
        selected_set = set(participants_list)

        upload_total = sum(int(update.get("upload_bytes") or 0) for update in updates)
        download_total = sum(int(update.get("download_bytes") or 0) for update in updates)
        raw_total = sum(int(update.get("raw_bytes") or 0) for update in updates)
        compressed_total = sum(int(update.get("compressed_bytes") or 0) for update in updates)
        compressed_ratio = (
            (raw_total / compressed_total) if compressed_total > 0 else 1.0
        )

        round_wall_time_ms = 0.0
        if self.round_start_time is not None:
            round_wall_time_ms = (time.time() - self.round_start_time) * 1000.0
        train_time_ms_avg = (
            sum(float(update.get("train_time_ms") or 0.0) for update in updates) / len(updates)
            if updates
            else 0.0
        )

        personal_acc = {
            str(update.get("client_id", "")): float(update.get("train_accuracy") or 0.0)
            for update in updates
        }
        acc_values = list(personal_acc.values())
        if acc_values:
            acc_avg = sum(acc_values) / len(acc_values)
            acc_var = sum((value - acc_avg) ** 2 for value in acc_values) / len(acc_values)
            acc_std = math.sqrt(acc_var)
            acc_min = min(acc_values)
            acc_sq_sum = sum(value ** 2 for value in acc_values)
            jain_index = (
                (sum(acc_values) ** 2) / (len(acc_values) * acc_sq_sum)
                if acc_sq_sum > 0
                else 0.0
            )
        else:
            acc_avg = 0.0
            acc_std = 0.0
            acc_min = 0.0
            jain_index = 0.0

        alerts = []
        excluded_reasons = {}
        cosine_scores = {}
        similarity_rank = []
        cosine_threshold = 0.0
        if detection:
            excluded_reasons = detection.get("excluded_reasons", {}) or {}
            cosine_scores = detection.get("cosine_scores", {}) or {}
            similarity_rank = detection.get("similarity_rank", []) or []
            cosine_threshold = float(detection.get("cosine_threshold") or 0.0)
        for client_id in excluded:
            reason_parts = excluded_reasons.get(client_id, [])
            if reason_parts:
                reason_text = "anomalous " + "/".join(reason_parts)
            else:
                reason_text = "anomalous loss or update norm"
            alerts.append(
                {
                    "time": time.strftime("%H:%M:%S"),
                    "round_id": self.current_round,
                    "client_id": client_id,
                    "reason": reason_text,
                    "action": "excluded",
                }
            )
        for client_id in dropped_clients:
            alerts.append(
                {
                    "time": time.strftime("%H:%M:%S"),
                    "round_id": self.current_round,
                    "client_id": client_id,
                    "reason": "timeout",
                    "action": "timeout",
                }
            )

        online_clients = self.client_provider() if self.client_provider else []
        if online_clients:
            for client in online_clients:
                client_id = str(client.get("client_id", ""))
                client["cooldown"] = self._is_in_cooldown(client_id)
        eligible_clients = self._eligible_clients()
        online_count = len(online_clients)
        if online_clients and "online" in online_clients[0]:
            online_count = sum(1 for client in online_clients if client.get("online"))
        client_counts = {
            "online": online_count,
            "eligible": len(eligible_clients),
            "registered": len(self.clients),
            "blacklisted": 0,
        }
        if self.client_stats_provider is not None:
            client_counts.update(self.client_stats_provider())

        client_updates = []
        excluded_set = set(excluded)
        for update in all_updates:
            client_id = str(update.get("client_id", ""))
            status = "excluded" if client_id in excluded_set else "included"
            if selected_set and client_id not in selected_set and client_id not in excluded_set:
                status = "not_selected"
            client_updates.append(
                {
                    "client_id": client_id,
                    "client_name": (
                        self.client_name_lookup(client_id) if self.client_name_lookup else ""
                    ),
                    "train_loss": float(update.get("train_loss", 0.0)),
                    "train_accuracy": float(update.get("train_accuracy", 0.0)),
                    "epsilon": float(update.get("epsilon", 0.0) or 0.0),
                    "upload_bytes": int(update.get("upload_bytes") or 0),
                    "status": status,
                    "label_histogram": update.get("label_histogram", {}),
                    "pre_dp_norm": update.get("pre_dp_norm"),
                    "clip_applied": update.get("clip_applied"),
                }
            )
        if dropped_clients:
            seen_ids = {entry.get("client_id") for entry in client_updates}
            for client_id in dropped_clients:
                if client_id in seen_ids:
                    continue
                client_updates.append(
                    {
                        "client_id": client_id,
                        "client_name": (
                            self.client_name_lookup(client_id)
                            if self.client_name_lookup
                            else ""
                        ),
                        "train_loss": None,
                        "train_accuracy": None,
                        "epsilon": None,
                        "upload_bytes": None,
                        "status": "timeout",
                        "label_histogram": {},
                        "pre_dp_norm": None,
                        "clip_applied": None,
                    }
                )

        malicious_clients = {
            str(update.get("client_id", ""))
            for update in all_updates
            if str(update.get("attack_type") or "").lower() not in ("", "none")
        }
        attack_types = {
            str(update.get("attack_type") or "").lower()
            for update in all_updates
            if str(update.get("attack_type") or "").lower() not in ("", "none")
        }
        detected_set = set(excluded)
        true_positive = len(detected_set & malicious_clients)
        precision = true_positive / len(detected_set) if detected_set else 0.0
        recall = true_positive / len(malicious_clients) if malicious_clients else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        attack_method = "none"
        if len(attack_types) == 1:
            attack_method = next(iter(attack_types))
        elif attack_types:
            attack_method = "mixed"

        sampling_info: Dict[str, object] = {
            "enabled": self.sampling_enabled,
            "strategy": self.sampling_strategy,
            "selection_mode": self._round_selection_mode,
            "epsilon": self.sampling_epsilon,
            "fairness_window": self.sampling_fairness_window,
        }
        if self.sampling_state_provider is not None:
            state = self.sampling_state_provider()
            selection_histogram = {
                client_id: int(info.get("selected_cnt", 0)) for client_id, info in state.items()
            }
            scores = {
                client_id: float(info.get("score", 0.0)) for client_id, info in state.items()
            }
            timeout_total = sum(float(info.get("timeout_cnt", 0.0)) for info in state.values())
            timeout_clients = sum(
                1 for info in state.values() if float(info.get("timeout_cnt", 0.0)) > 0.0
            )
            selected_total = sum(float(info.get("selected_cnt", 0.0)) for info in state.values())
            timeout_rate = timeout_total / selected_total if selected_total > 0 else 0.0
            score_rank = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]
            sampling_info.update(
                {
                    "selection_histogram": selection_histogram,
                    "timeout_rate": timeout_rate,
                    "timeout_clients": timeout_clients,
                    "score_rank": score_rank,
                }
            )

        metric = {
            "round_id": self.current_round,
            "global_loss": round_loss,
            "global_accuracy": round_accuracy,
            "participants": participants_list,
            "dropped_clients": dropped_clients,
            "privacy": {
                "enabled": bool(self.dp_config.get("enabled", False)),
                "mode": dp_mode,
                "epsilon_accountant": epsilon_accountant,
                "target_epsilon": self.dp_target_epsilon,
                "epsilon_max": epsilon_max,
                "epsilon_avg": epsilon_avg,
                "delta": self.dp_config.get("delta"),
                "noise_multiplier": self._current_dp_params.get(
                    "noise_multiplier", self.dp_noise_multiplier
                ),
                "clip_norm": self._current_dp_params.get("clip_norm", self.dp_clip_norm),
                "clip_rate": clip_rate,
            },
            "comm": {
                "upload_bytes_total": upload_total,
                "download_bytes_total": download_total,
                "compressed_ratio": compressed_ratio,
            },
            "robust": {
                "agg_method": self.robust_agg_method,
                "excluded_clients": excluded,
            },
            "timing": {
                "round_wall_time_ms": round_wall_time_ms,
                "train_time_ms_avg": train_time_ms_avg,
            },
            "fairness": {
                "client_personal_acc": personal_acc,
                "avg_accuracy": acc_avg,
                "min_accuracy": acc_min,
                "std_accuracy": acc_std,
                "jain_index": jain_index,
            },
            "alerts": alerts,
            "online_clients": online_clients,
            "client_counts": client_counts,
            "client_updates": client_updates,
            "server": {
                "eval_loss": eval_loss,
                "eval_accuracy": eval_accuracy,
                "update_norm": update_norm,
            },
            "security": {
                "attack_simulation": {
                    "enabled": bool(malicious_clients),
                    "method": attack_method,
                    "malicious_count": len(malicious_clients),
                },
                "detector": {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "cosine_threshold": cosine_threshold,
                    "true_positive": true_positive,
                    "detected": len(detected_set),
                    "malicious": len(malicious_clients),
                },
                "similarity_rank": similarity_rank,
                "cosine_scores": cosine_scores,
            },
            "sampling": sampling_info,
            "deadline_triggered": deadline_triggered,
        }
        self.metrics_agent.record(metric)
