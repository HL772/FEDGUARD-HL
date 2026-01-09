import os
import time
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from server.client_manager import ClientManagerAgent
from server.metrics.store import MetricsAgent
from server.orchestrator import CoordinatorAgent
from client.compression.topk import decompress_state


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

app = FastAPI()

CONFIG = _load_config(os.environ.get("CONFIG_PATH", "experiments/configs/default.yaml"))
SECURITY_CFG = CONFIG.get("security", {})
DP_CFG = CONFIG.get("dp", {})
COMPRESSION_CFG = CONFIG.get("compression", {})
TRAIN_CFG = CONFIG.get("train", {})
MODEL_CFG = CONFIG.get("model", {})
SAMPLING_CFG = CONFIG.get("sampling", {})
COSINE_CFG = SECURITY_CFG.get("cosine_detection", {})
ONLINE_TTL_SEC = float(os.environ.get("ONLINE_TTL_SEC", "60"))

CLIENT_MANAGER = ClientManagerAgent(online_ttl_sec=ONLINE_TTL_SEC)
ELIGIBLE_PROVIDER = CLIENT_MANAGER.eligible_client_ids

METRICS = MetricsAgent("server/metrics/metrics.jsonl", clear_on_start=True)

COORDINATOR = CoordinatorAgent(
    max_rounds=int(os.environ.get("MAX_ROUNDS", "3")),
    clients_per_round=int(os.environ.get("CLIENTS_PER_ROUND", "3")),
    lr=float(os.environ.get("LR", "0.1")),
    batch_size=int(os.environ.get("BATCH_SIZE", "32")),
    epochs=int(os.environ.get("EPOCHS", "1")),
    secure_aggregation=bool(SECURITY_CFG.get("secure_aggregation", False)),
    malicious_detection=bool(SECURITY_CFG.get("malicious_detection", True)),
    robust_agg_method=str(SECURITY_CFG.get("robust_aggregation", "fedavg")),
    trim_ratio=float(SECURITY_CFG.get("trim_ratio", 0.2)),
    loss_threshold=float(SECURITY_CFG.get("loss_threshold", 3.0)),
    norm_threshold=float(SECURITY_CFG.get("norm_threshold", 3.0)),
    require_both=bool(SECURITY_CFG.get("require_both", True)),
    min_mad=float(SECURITY_CFG.get("min_mad", 0.05)),
    cosine_threshold=float(COSINE_CFG.get("threshold", 2.5)),
    cosine_enabled=bool(COSINE_CFG.get("enabled", False)),
    cosine_top_k=int(COSINE_CFG.get("top_k", 5)),
    metrics_agent=METRICS,
    dp_config=DP_CFG,
    compression_config=COMPRESSION_CFG,
    client_provider=CLIENT_MANAGER.online_clients,
    client_name_lookup=CLIENT_MANAGER.get_client_name,
    eligible_clients_provider=ELIGIBLE_PROVIDER,
    exclude_callback=lambda client_ids: CLIENT_MANAGER.blacklist_clients(
        client_ids, reason="anomalous loss or update norm"
    ),
    client_stats_provider=CLIENT_MANAGER.stats,
    fedprox_mu=float(TRAIN_CFG.get("fedprox_mu", 0.0)),
    deadline_ms=float(TRAIN_CFG.get("deadline_ms", 0.0)),
    train_algo=str(TRAIN_CFG.get("algo", "fedavg")),
    model_split=MODEL_CFG.get("split", {}),
    public_head_eval=MODEL_CFG.get("public_head_eval", {}),
    sampling_config=SAMPLING_CFG,
    sampling_state_provider=CLIENT_MANAGER.sampling_state,
    sampling_record_selected=CLIENT_MANAGER.record_selected,
    sampling_record_timeout=CLIENT_MANAGER.record_timeouts,
    sampling_update_scores=CLIENT_MANAGER.update_scores,
    byzantine_f=int(SECURITY_CFG.get("byzantine_f", 1)),
    server_lr=float(TRAIN_CFG.get("server_lr", 1.0)),
    server_update_clip=float(TRAIN_CFG.get("server_update_clip", 0.0)),
    timeout_simulation=TRAIN_CFG.get("timeout_simulation", {}),
)

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.on_event("startup")
async def _startup() -> None:
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


@app.post("/api/v1/join", response_model=JoinResponse)
def join(req: JoinRequest) -> JoinResponse:
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
    return FileResponse(str(WEB_DIR / "dashboard.html"))


@app.get("/api/v1/metrics/latest")
def metrics_latest() -> Dict[str, Any]:
    return METRICS.latest() or {}


@app.get("/api/v1/metrics/all")
def metrics_all() -> Dict[str, Any]:
    return {"metrics": METRICS.all()}


@app.get("/api/v1/clients")
def clients_list() -> Dict[str, Any]:
    return {"clients": CLIENT_MANAGER.online_clients()}


@app.post("/api/v1/clients/blacklist", response_model=ClientActionResponse)
def clients_blacklist(req: ClientActionRequest) -> ClientActionResponse:
    if not CLIENT_MANAGER.is_registered(req.client_id):
        raise HTTPException(status_code=404, detail="client not found")
    CLIENT_MANAGER.blacklist_clients([req.client_id], reason=req.reason or "manual")
    return ClientActionResponse(status="ok", client_id=req.client_id, blacklisted=True)


@app.post("/api/v1/clients/unblacklist", response_model=ClientActionResponse)
def clients_unblacklist(req: ClientActionRequest) -> ClientActionResponse:
    if not CLIENT_MANAGER.is_registered(req.client_id):
        raise HTTPException(status_code=404, detail="client not found")
    CLIENT_MANAGER.unblacklist_clients([req.client_id])
    return ClientActionResponse(status="ok", client_id=req.client_id, blacklisted=False)


@app.post("/api/v1/get_model", response_model=GetModelResponse)
def get_model(req: GetModelRequest) -> GetModelResponse:
    if not CLIENT_MANAGER.is_registered(req.client_id):
        raise HTTPException(status_code=404, detail="client not found")
    payload = COORDINATOR.get_round_payload(req.client_id)
    return GetModelResponse(**payload)


@app.post("/api/v1/submit_update", response_model=SubmitUpdateResponse)
def submit_update(req: SubmitUpdateRequest) -> SubmitUpdateResponse:
    if not CLIENT_MANAGER.is_registered(req.client_id):
        raise HTTPException(status_code=404, detail="client not found")
    masked_update = req.masked_update
    delta_state = req.delta_state
    if req.compressed_update is not None:
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
