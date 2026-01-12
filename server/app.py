import os
import sys
import time
import asyncio
import copy
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional, List

# 允许直接运行 python server/app.py
_APP_ROOT = Path(__file__).resolve().parents[1]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from server.client_manager import ClientManagerModule
from server.metrics.store import MetricsModule
from server.orchestrator import CoordinatorModule
from server.launcher import LocalDemoRunner
from client.compression.topk import decompress_state

# 服务端入口（课设要求对应点）：
# - REST API：join/heartbeat/get_model/submit_update
# - WebSocket 指标推送
# - Dashboard 静态页面托管（AGENT.md 2.x / 3.1 A/B/F/G）


def _parse_value(raw: str) -> Any:
    lower = raw.lower()
    if lower in ("true", "false"):
        return lower == "true"
    try:
        if "." in raw or "e" in lower:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _load_config(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
            return data or {}
    except Exception:
        config: Dict[str, Any] = {}
        current_section: Optional[str] = None
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                # Remove inline comments
                if "#" in line:
                    line = line.split("#", 1)[0].strip()
                
                if not line:
                    continue

                if line.endswith(":"):
                    current_section = line[:-1].strip()
                    config[current_section] = {}
                    continue
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    target = config[current_section] if current_section else config
                    target[key] = _parse_value(value)
        return config


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_config_path(path: Optional[str]) -> Path:
    root = _project_root()
    if not path:
        return root / "experiments" / "configs" / "default.yaml"
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    configs_dir = (root / "experiments" / "configs").resolve()
    if configs_dir not in candidate.parents and candidate != configs_dir:
        raise ValueError("config path not allowed")
    return candidate


def _list_configs() -> List[str]:
    root = _project_root()
    config_dir = root / "experiments" / "configs"
    if not config_dir.exists():
        return []
    return sorted(str(path.relative_to(root)) for path in config_dir.glob("*.yaml"))


def _apply_overrides(config: Dict[str, Any], req: "StartSessionRequest") -> Dict[str, Any]:
    merged = copy.deepcopy(config)
    dp_cfg = merged.setdefault("dp", {})
    if req.dp_enabled is not None:
        dp_cfg["enabled"] = bool(req.dp_enabled)
    if req.dp_mode:
        dp_cfg["mode"] = str(req.dp_mode)
    if req.dp_clip_norm is not None:
        dp_cfg["clip_norm"] = float(req.dp_clip_norm)
    if req.dp_noise_multiplier is not None:
        dp_cfg["noise_multiplier"] = float(req.dp_noise_multiplier)
    if req.dp_delta is not None:
        dp_cfg["delta"] = float(req.dp_delta)
    if req.dp_target_epsilon is not None:
        dp_cfg["target_epsilon"] = float(req.dp_target_epsilon)
    if req.dp_schedule_type is not None:
        dp_cfg.setdefault("schedule", {})["type"] = str(req.dp_schedule_type)
    if req.dp_sigma_end_ratio is not None:
        dp_cfg.setdefault("schedule", {})["sigma_end_ratio"] = float(req.dp_sigma_end_ratio)
    if req.dp_adaptive_clip_enabled is not None:
        dp_cfg.setdefault("adaptive_clip", {})["enabled"] = bool(req.dp_adaptive_clip_enabled)
    if req.dp_adaptive_clip_percentile is not None:
        dp_cfg.setdefault("adaptive_clip", {})["percentile"] = float(
            req.dp_adaptive_clip_percentile
        )
    if req.dp_adaptive_clip_ema is not None:
        dp_cfg.setdefault("adaptive_clip", {})["ema"] = float(req.dp_adaptive_clip_ema)

    comp_cfg = merged.setdefault("compression", {})
    if req.compression_enabled is not None:
        comp_cfg["enabled"] = bool(req.compression_enabled)
    if req.compression_topk_ratio is not None:
        comp_cfg["topk_ratio"] = float(req.compression_topk_ratio)
    if req.compression_quant_bits is not None:
        comp_cfg["quant_bits"] = int(req.compression_quant_bits)
    if req.compression_error_feedback is not None:
        comp_cfg["error_feedback"] = bool(req.compression_error_feedback)

    sec_cfg = merged.setdefault("security", {})
    if req.secure_aggregation is not None:
        sec_cfg["secure_aggregation"] = bool(req.secure_aggregation)
    if req.malicious_detection is not None:
        sec_cfg["malicious_detection"] = bool(req.malicious_detection)
    if req.robust_aggregation:
        sec_cfg["robust_aggregation"] = str(req.robust_aggregation)
    if req.trim_ratio is not None:
        sec_cfg["trim_ratio"] = float(req.trim_ratio)
    if req.byzantine_f is not None:
        sec_cfg["byzantine_f"] = int(req.byzantine_f)
    if req.loss_threshold is not None:
        sec_cfg["loss_threshold"] = float(req.loss_threshold)
    if req.norm_threshold is not None:
        sec_cfg["norm_threshold"] = float(req.norm_threshold)
    if req.require_both is not None:
        sec_cfg["require_both"] = bool(req.require_both)
    if req.min_mad is not None:
        sec_cfg["min_mad"] = float(req.min_mad)
    cos_cfg = sec_cfg.setdefault("cosine_detection", {})
    if req.cosine_enabled is not None:
        cos_cfg["enabled"] = bool(req.cosine_enabled)
    if req.cosine_threshold is not None:
        cos_cfg["threshold"] = float(req.cosine_threshold)
    if req.cosine_top_k is not None:
        cos_cfg["top_k"] = int(req.cosine_top_k)
    attack_cfg = sec_cfg.setdefault("attack_simulation", {})
    if req.attack_enabled is not None:
        attack_cfg["enabled"] = bool(req.attack_enabled)
    if req.attack_method:
        attack_cfg["method"] = str(req.attack_method)
    if req.attack_scale is not None:
        attack_cfg["scale"] = float(req.attack_scale)
    if req.attack_malicious_fraction is not None:
        attack_cfg["malicious_fraction"] = float(req.attack_malicious_fraction)
    if req.attack_malicious_ranks is not None:
        attack_cfg["malicious_ranks"] = [int(v) for v in req.attack_malicious_ranks]
    if req.attack_label_flip is not None:
        attack_cfg["label_flip"] = bool(req.attack_label_flip)
    if req.attack_loss_scale is not None:
        attack_cfg["loss_scale"] = float(req.attack_loss_scale)
    if req.attack_accuracy_scale is not None:
        attack_cfg["accuracy_scale"] = float(req.attack_accuracy_scale)

    train_cfg = merged.setdefault("train", {})
    if req.train_algo:
        train_cfg["algo"] = str(req.train_algo)
    if req.train_lr is not None:
        train_cfg["lr"] = float(req.train_lr)
    if req.train_epochs is not None:
        train_cfg["epochs"] = int(req.train_epochs)
    if req.fedprox_mu is not None:
        train_cfg["fedprox_mu"] = float(req.fedprox_mu)
    if req.deadline_ms is not None:
        train_cfg["deadline_ms"] = float(req.deadline_ms)
    if req.server_lr is not None:
        train_cfg["server_lr"] = float(req.server_lr)
    if req.server_update_clip is not None:
        train_cfg["server_update_clip"] = float(req.server_update_clip)
    if req.private_head_lr is not None:
        train_cfg["private_head_lr"] = float(req.private_head_lr)
    if req.private_head_epochs is not None:
        train_cfg["private_head_epochs"] = int(req.private_head_epochs)

    data_cfg = merged.setdefault("data", {})
    if req.data_alpha is not None:
        data_cfg["alpha"] = float(req.data_alpha)

    sampling_cfg = merged.setdefault("sampling", {})
    if req.sampling_enabled is not None:
        sampling_cfg["enabled"] = bool(req.sampling_enabled)
    if req.sampling_strategy:
        sampling_cfg["strategy"] = str(req.sampling_strategy)
    if req.sampling_epsilon is not None:
        sampling_cfg["epsilon"] = float(req.sampling_epsilon)
    if req.sampling_score_ema is not None:
        sampling_cfg["score_ema"] = float(req.sampling_score_ema)
    if req.sampling_timeout_penalty is not None:
        sampling_cfg["timeout_penalty"] = float(req.sampling_timeout_penalty)
    if req.sampling_anomaly_penalty is not None:
        sampling_cfg["anomaly_penalty"] = float(req.sampling_anomaly_penalty)
    if req.sampling_fairness_window is not None:
        sampling_cfg["fairness_window"] = int(req.sampling_fairness_window)
    if req.sampling_softmax_temp is not None:
        sampling_cfg["softmax_temp"] = float(req.sampling_softmax_temp)

    timeout_cfg = train_cfg.setdefault("timeout_simulation", {})
    if req.timeout_simulation_enabled is not None:
        timeout_cfg["enabled"] = bool(req.timeout_simulation_enabled)
    if req.timeout_client_ranks is not None:
        timeout_cfg["client_ranks"] = [int(v) for v in req.timeout_client_ranks]
    if req.timeout_cooldown_rounds is not None:
        timeout_cfg["cooldown_rounds"] = int(req.timeout_cooldown_rounds)
    return merged


def _write_runtime_config(config: Dict[str, Any], base_name: str) -> str:
    root = _project_root()
    runtime_dir = root / "experiments" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_dir / f"web_{base_name}.yaml"
    try:
        import yaml  # type: ignore

        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    except Exception:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
    return str(path)


def _default_server_url() -> str:
    host = os.environ.get("SERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("SERVER_API_PORT", "8000"))
    if host in ("0.0.0.0", "::", "[::]"):
        host = "127.0.0.1"
    return f"http://{host}:{port}"

app = FastAPI()

PROJECT_ROOT = _project_root()

DEFAULT_CONFIG_PATH = _resolve_config_path(
    os.environ.get("CONFIG_PATH", "experiments/configs/default.yaml")
).as_posix()
CONFIG = _load_config(DEFAULT_CONFIG_PATH)
ONLINE_TTL_SEC = float(os.environ.get("ONLINE_TTL_SEC", "600"))

CLIENT_MANAGER = ClientManagerModule(online_ttl_sec=ONLINE_TTL_SEC)

# MetricsModule：轮次指标持久化与实时推送（性能监控）
metrics_clear_on_start = os.environ.get("METRICS_CLEAR_ON_START", "0") == "1"
METRICS = MetricsModule("server/metrics/metrics.jsonl", clear_on_start=metrics_clear_on_start)


def _build_coordinator(
    config: Dict[str, Any],
    *,
    max_rounds: int,
    clients_per_round: int,
    online_ttl_sec: float,
) -> CoordinatorModule:
    security_cfg = config.get("security", {})
    dp_cfg = config.get("dp", {})
    compression_cfg = config.get("compression", {})
    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})
    sampling_cfg = config.get("sampling", {})
    cosine_cfg = security_cfg.get("cosine_detection", {})

    global CLIENT_MANAGER
    CLIENT_MANAGER = ClientManagerModule(online_ttl_sec=online_ttl_sec)
    eligible_provider = CLIENT_MANAGER.eligible_client_ids

    return CoordinatorModule(
        max_rounds=max_rounds,
        clients_per_round=clients_per_round,
        lr=float(os.environ.get("LR", train_cfg.get("lr", 0.1))),
        batch_size=int(os.environ.get("BATCH_SIZE", "32")),
        epochs=int(os.environ.get("EPOCHS", train_cfg.get("epochs", 1))),
        secure_aggregation=bool(security_cfg.get("secure_aggregation", False)),
        malicious_detection=bool(security_cfg.get("malicious_detection", True)),
        robust_agg_method=str(security_cfg.get("robust_aggregation", "fedavg")),
        trim_ratio=float(security_cfg.get("trim_ratio", 0.2)),
        loss_threshold=float(security_cfg.get("loss_threshold", 3.0)),
        norm_threshold=float(security_cfg.get("norm_threshold", 3.0)),
        require_both=bool(security_cfg.get("require_both", True)),
        min_mad=float(security_cfg.get("min_mad", 0.05)),
        cosine_threshold=float(cosine_cfg.get("threshold", 2.5)),
        cosine_enabled=bool(cosine_cfg.get("enabled", False)),
        cosine_top_k=int(cosine_cfg.get("top_k", 5)),
        metrics_agent=METRICS,
        dp_config=dp_cfg,
        compression_config=compression_cfg,
        client_provider=CLIENT_MANAGER.all_clients,
        client_name_lookup=CLIENT_MANAGER.get_client_name,
        eligible_clients_provider=eligible_provider,
        exclude_callback=lambda client_ids: CLIENT_MANAGER.blacklist_clients(
            client_ids, reason="anomalous loss or update norm"
        ),
        client_stats_provider=CLIENT_MANAGER.stats,
        fedprox_mu=float(train_cfg.get("fedprox_mu", 0.0)),
        deadline_ms=float(train_cfg.get("deadline_ms", 0.0)),
        train_algo=str(train_cfg.get("algo", "fedavg")),
        model_split=model_cfg.get("split", {}),
        public_head_eval=model_cfg.get("public_head_eval", {}),
        sampling_config=sampling_cfg,
        sampling_state_provider=CLIENT_MANAGER.sampling_state,
        sampling_record_selected=CLIENT_MANAGER.record_selected,
        sampling_record_timeout=CLIENT_MANAGER.record_timeouts,
        sampling_update_scores=CLIENT_MANAGER.update_scores,
        byzantine_f=int(security_cfg.get("byzantine_f", 1)),
        server_lr=float(train_cfg.get("server_lr", 1.0)),
        server_update_clip=float(train_cfg.get("server_update_clip", 0.0)),
        timeout_simulation=train_cfg.get("timeout_simulation", {}),
    )


COORDINATOR = _build_coordinator(
    CONFIG,
    max_rounds=int(os.environ.get("MAX_ROUNDS", "3")),
    clients_per_round=int(os.environ.get("CLIENTS_PER_ROUND", "3")),
    online_ttl_sec=ONLINE_TTL_SEC,
)

SESSION_LOCK = threading.Lock()
RUNNER = LocalDemoRunner(PROJECT_ROOT)
SESSION_STATE: Dict[str, Any] = {
    "config_path": DEFAULT_CONFIG_PATH,
    "runtime_config_path": DEFAULT_CONFIG_PATH,
    "params": {},
}

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.on_event("startup")
async def _startup() -> None:
    # 绑定事件循环用于 WebSocket 广播调度
    METRICS.set_event_loop(asyncio.get_running_loop())


class JoinRequest(BaseModel):
    client_name: Optional[str] = None


class JoinResponse(BaseModel):
    client_id: str


class HeartbeatRequest(BaseModel):
    client_id: str
    timestamp: Optional[float] = None


class HeartbeatResponse(BaseModel):
    status: str
    server_time: float


class GetModelRequest(BaseModel):
    client_id: str


class GetModelResponse(BaseModel):
    status: str
    round_id: Optional[int] = None
    model_state: Optional[Dict[str, list]] = None
    config: Optional[Dict[str, object]] = None
    reason: Optional[str] = None
    participants: Optional[list[str]] = None


class ClientActionRequest(BaseModel):
    client_id: str
    reason: Optional[str] = None


class ClientActionResponse(BaseModel):
    status: str
    client_id: str
    blacklisted: bool


class SubmitUpdateRequest(BaseModel):
    client_id: str
    round_id: int
    model_state: Optional[Dict[str, list]] = None
    delta_state: Optional[Dict[str, list]] = None
    masked_update: Optional[Dict[str, list]] = None
    compressed_update: Optional[Dict[str, object]] = None
    compressed_type: Optional[str] = None
    num_samples: int
    train_loss: float
    epsilon: Optional[float] = None
    cosine_score: Optional[float] = None
    update_norm: Optional[float] = None
    train_accuracy: Optional[float] = None
    upload_bytes: Optional[int] = None
    download_bytes: Optional[int] = None
    raw_bytes: Optional[int] = None
    compressed_bytes: Optional[int] = None
    train_time_ms: Optional[float] = None
    label_histogram: Optional[Dict[str, int]] = None
    train_steps: Optional[int] = None
    pre_dp_norm: Optional[float] = None
    clip_applied: Optional[bool] = None
    attack_type: Optional[str] = None


class SubmitUpdateResponse(BaseModel):
    status: str
    reason: Optional[str] = None
    current_round: Optional[int] = None


class StartSessionRequest(BaseModel):
    num_clients: int = 14
    clients_per_round: Optional[int] = 8
    client_batch_size: int = 14
    max_rounds: int = 20
    server_url: Optional[str] = None
    online_ttl_sec: Optional[float] = None
    dp_enabled: Optional[bool] = None
    dp_mode: Optional[str] = None
    dp_clip_norm: Optional[float] = None
    dp_noise_multiplier: Optional[float] = None
    dp_delta: Optional[float] = None
    dp_target_epsilon: Optional[float] = None
    dp_schedule_type: Optional[str] = None
    dp_sigma_end_ratio: Optional[float] = None
    dp_adaptive_clip_enabled: Optional[bool] = None
    dp_adaptive_clip_percentile: Optional[float] = None
    dp_adaptive_clip_ema: Optional[float] = None

    compression_enabled: Optional[bool] = None
    compression_topk_ratio: Optional[float] = None
    compression_quant_bits: Optional[int] = None
    compression_error_feedback: Optional[bool] = None

    secure_aggregation: Optional[bool] = None
    malicious_detection: Optional[bool] = None
    robust_aggregation: Optional[str] = None
    trim_ratio: Optional[float] = None
    byzantine_f: Optional[int] = None
    loss_threshold: Optional[float] = None
    norm_threshold: Optional[float] = None
    require_both: Optional[bool] = None
    min_mad: Optional[float] = None
    cosine_enabled: Optional[bool] = None
    cosine_threshold: Optional[float] = None
    cosine_top_k: Optional[int] = None

    attack_enabled: Optional[bool] = None
    attack_method: Optional[str] = None
    attack_scale: Optional[float] = None
    attack_malicious_fraction: Optional[float] = None
    attack_malicious_ranks: Optional[List[int]] = None
    attack_label_flip: Optional[bool] = None
    attack_loss_scale: Optional[float] = None
    attack_accuracy_scale: Optional[float] = None

    train_algo: Optional[str] = None
    train_lr: Optional[float] = None
    train_epochs: Optional[int] = None
    fedprox_mu: Optional[float] = None
    deadline_ms: Optional[float] = None
    server_lr: Optional[float] = None
    server_update_clip: Optional[float] = None
    private_head_lr: Optional[float] = None
    private_head_epochs: Optional[int] = None

    data_alpha: Optional[float] = None

    sampling_enabled: Optional[bool] = None
    sampling_strategy: Optional[str] = None
    sampling_epsilon: Optional[float] = None
    sampling_score_ema: Optional[float] = None
    sampling_timeout_penalty: Optional[float] = None
    sampling_anomaly_penalty: Optional[float] = None
    sampling_fairness_window: Optional[int] = None
    sampling_softmax_temp: Optional[float] = None

    timeout_simulation_enabled: Optional[bool] = None
    timeout_client_ranks: Optional[List[int]] = None
    timeout_cooldown_rounds: Optional[int] = None

    download_data: Optional[bool] = None
    stay_online_on_not_selected: Optional[bool] = None


class SessionStatusResponse(BaseModel):
    running: bool
    status: Optional[str] = None
    config_path: Optional[str] = None
    runtime_config_path: Optional[str] = None
    params: Dict[str, object] = {}
    last_error: Optional[str] = None


@app.post("/api/v1/join", response_model=JoinResponse)
def join(req: JoinRequest) -> JoinResponse:
    # 客户端注册（ClientManagerModule 职责）
    client_id, reused = CLIENT_MANAGER.register(req.client_name)
    COORDINATOR.register_client(client_id)
    if reused:
        print(
            f"[server] rejoin client_id={client_id} name={req.client_name or ''}",
            flush=True,
        )
    else:
        print(
            f"[server] join client_id={client_id} name={req.client_name or ''}",
            flush=True,
        )
    return JoinResponse(client_id=client_id)


@app.post("/api/v1/heartbeat", response_model=HeartbeatResponse)
def heartbeat(req: HeartbeatRequest) -> HeartbeatResponse:
    # 心跳用于在线状态维护与可参与资格判断
    if not CLIENT_MANAGER.heartbeat(req.client_id, req.timestamp):
        raise HTTPException(status_code=404, detail="client not found")
    now = time.time()
    print(
        f"[server] heartbeat client_id={req.client_id} ts={req.timestamp or 0.0:.2f}",
        flush=True,
    )
    return HeartbeatResponse(status="ok", server_time=now)


@app.get("/dashboard")
def dashboard() -> FileResponse:
    # Dashboard 页面（WebDashboardModule）
    return FileResponse(str(WEB_DIR / "dashboard.html"))


@app.get("/api/v1/metrics/latest")
def metrics_latest() -> Dict[str, Any]:
    # 最新轮次指标（轮询兜底）
    return METRICS.latest() or {}


@app.get("/api/v1/metrics/all")
def metrics_all() -> Dict[str, Any]:
    # 历史指标（页面初始化时加载）
    return {"metrics": METRICS.all()}


@app.get("/api/v1/clients")
def clients_list() -> Dict[str, Any]:
    # 在线客户端列表（仪表盘 + 手动管理）
    return {"clients": CLIENT_MANAGER.all_clients()}


@app.get("/api/v1/configs")
def configs_list() -> Dict[str, Any]:
    # 可选配置列表（用于网页下拉选择）
    return {"configs": _list_configs(), "default": "experiments/configs/default.yaml"}


@app.get("/api/v1/config/default")
def config_default() -> Dict[str, Any]:
    # 返回默认配置（页面初始化用）
    num_clients = int(os.environ.get("NUM_CLIENTS", "14"))
    return {
        "path": DEFAULT_CONFIG_PATH,
        "config": _load_config(DEFAULT_CONFIG_PATH),
        "runtime": {
            "online_ttl_sec": ONLINE_TTL_SEC,
            "max_rounds": int(os.environ.get("MAX_ROUNDS", "20")),
            "num_clients": num_clients,
            "clients_per_round": int(os.environ.get("CLIENTS_PER_ROUND", "8")),
            "client_batch_size": int(
                os.environ.get("CLIENT_BATCH_SIZE", str(num_clients))
            ),
        },
    }


@app.get("/api/v1/session/status", response_model=SessionStatusResponse)
def session_status() -> SessionStatusResponse:
    # 返回当前训练会话状态
    return SessionStatusResponse(
        running=RUNNER.is_running(),
        status=RUNNER.status(),
        config_path=SESSION_STATE.get("config_path"),
        runtime_config_path=SESSION_STATE.get("runtime_config_path"),
        params=SESSION_STATE.get("params", {}),
        last_error=RUNNER.last_error or None,
    )


@app.post("/api/v1/session/start", response_model=SessionStatusResponse)
def session_start(req: StartSessionRequest) -> SessionStatusResponse:
    # 根据网页配置启动训练（启动客户端进程）
    with SESSION_LOCK:
        if RUNNER.is_running():
            raise HTTPException(status_code=409, detail="session already running")
        config_path = DEFAULT_CONFIG_PATH
        base_config = _load_config(DEFAULT_CONFIG_PATH)
        merged_config = _apply_overrides(base_config, req)
        runtime_path = _write_runtime_config(merged_config, Path(config_path).stem)

        max_rounds = max(int(req.max_rounds), 1)
        num_clients = max(int(req.num_clients), 1)
        clients_per_round = int(req.clients_per_round or num_clients)
        client_batch_size = max(int(req.client_batch_size), 1)
        online_ttl_sec = float(req.online_ttl_sec or ONLINE_TTL_SEC)
        server_url = (req.server_url or "").strip() or _default_server_url()

        if os.environ.get("METRICS_RESET_ON_SESSION_START", "1") == "1":
            METRICS.reset(clear_file=True)
        global COORDINATOR
        COORDINATOR = _build_coordinator(
            merged_config,
            max_rounds=max_rounds,
            clients_per_round=clients_per_round,
            online_ttl_sec=online_ttl_sec,
        )

        started = RUNNER.start(
            num_clients=num_clients,
            client_batch_size=client_batch_size,
            max_rounds=max_rounds,
            config_path=runtime_path,
            server_url=server_url,
            download_data=bool(req.download_data),
            stay_online_on_not_selected=bool(req.stay_online_on_not_selected),
        )
        if not started:
            raise HTTPException(status_code=409, detail="session already running")

        SESSION_STATE.update(
            {
                "config_path": config_path,
                "runtime_config_path": runtime_path,
                "params": {
                    "num_clients": num_clients,
                    "clients_per_round": clients_per_round,
                    "client_batch_size": client_batch_size,
                    "max_rounds": max_rounds,
                    "online_ttl_sec": online_ttl_sec,
                    "server_url": server_url,
                    "dp_enabled": req.dp_enabled,
                    "dp_mode": req.dp_mode,
                    "train_algo": req.train_algo,
                },
            }
        )
        return session_status()


@app.post("/api/v1/session/stop", response_model=SessionStatusResponse)
def session_stop() -> SessionStatusResponse:
    # 停止当前训练会话（终止客户端进程）
    with SESSION_LOCK:
        RUNNER.stop()
        COORDINATOR.training_done = True
    return session_status()


@app.post("/api/v1/clients/blacklist", response_model=ClientActionResponse)
def clients_blacklist(req: ClientActionRequest) -> ClientActionResponse:
    # 手动拉黑（客户端管理）
    if not CLIENT_MANAGER.is_registered(req.client_id):
        raise HTTPException(status_code=404, detail="client not found")
    CLIENT_MANAGER.blacklist_clients([req.client_id], reason=req.reason or "manual")
    return ClientActionResponse(status="ok", client_id=req.client_id, blacklisted=True)


@app.post("/api/v1/clients/unblacklist", response_model=ClientActionResponse)
def clients_unblacklist(req: ClientActionRequest) -> ClientActionResponse:
    # 手动解除拉黑（客户端管理）
    if not CLIENT_MANAGER.is_registered(req.client_id):
        raise HTTPException(status_code=404, detail="client not found")
    CLIENT_MANAGER.unblacklist_clients([req.client_id])
    return ClientActionResponse(status="ok", client_id=req.client_id, blacklisted=False)


@app.post("/api/v1/get_model", response_model=GetModelResponse)
def get_model(req: GetModelRequest) -> GetModelResponse:
    # 下发模型参数 + 轮次配置（CoordinatorModule 输出）
    if not CLIENT_MANAGER.is_registered(req.client_id):
        raise HTTPException(status_code=404, detail="client not found")
    # 任何有效请求都视为在线活跃
    CLIENT_MANAGER.heartbeat(req.client_id, time.time())
    payload = COORDINATOR.get_round_payload(req.client_id)
    return GetModelResponse(**payload)


@app.post("/api/v1/submit_update", response_model=SubmitUpdateResponse)
def submit_update(req: SubmitUpdateRequest) -> SubmitUpdateResponse:
    # 客户端更新提交（DP/压缩/掩码均在客户端完成）
    if not CLIENT_MANAGER.is_registered(req.client_id):
        raise HTTPException(status_code=404, detail="client not found")
    # 提交更新时刷新在线状态，避免长轮次误判离线
    CLIENT_MANAGER.heartbeat(req.client_id, time.time())
    masked_update = req.masked_update
    delta_state = req.delta_state
    if req.compressed_update is not None:
        # 压缩更新先解压，再交给聚合逻辑
        decompressed = decompress_state(req.compressed_update)
        if req.compressed_type == "masked":
            masked_update = decompressed
        else:
            delta_state = decompressed
    model_state = req.model_state
    if delta_state is None and model_state is None and masked_update is None:
        raise HTTPException(status_code=400, detail="missing update payload")
    result = COORDINATOR.submit_update(
        req.client_id,
        req.round_id,
        model_state=model_state,
        delta_state=delta_state,
        masked_update=masked_update,
        num_samples=req.num_samples,
        train_loss=req.train_loss,
        update_norm=req.update_norm,
        train_accuracy=req.train_accuracy,
        epsilon=req.epsilon,
        cosine_score=req.cosine_score,
        upload_bytes=req.upload_bytes,
        download_bytes=req.download_bytes,
        raw_bytes=req.raw_bytes,
        compressed_bytes=req.compressed_bytes,
        train_time_ms=req.train_time_ms,
        label_histogram=req.label_histogram,
        train_steps=req.train_steps,
        pre_dp_norm=req.pre_dp_norm,
        clip_applied=req.clip_applied,
        attack_type=req.attack_type,
    )
    return SubmitUpdateResponse(**result)


@app.websocket("/ws/metrics")
async def ws_metrics(websocket: WebSocket) -> None:
    # 指标推送（Dashboard 实时曲线）
    await METRICS.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        METRICS.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("SERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("SERVER_API_PORT", "8000"))
    print(f"[server] starting on {host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")
