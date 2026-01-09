import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class ApiError(RuntimeError):
    pass


def _request_json(method: str, url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8") if exc.fp else str(exc)
        raise ApiError(f"HTTP {exc.code} {exc.reason}: {detail}") from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        raise ApiError(f"Request failed: {reason}") from exc

    if not body:
        return {}
    return json.loads(body)


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def join(self, client_name: Optional[str] = None) -> str:
        payload: Dict[str, Any] = {}
        if client_name:
            payload["client_name"] = client_name
        resp = _request_json("POST", f"{self.base_url}/api/v1/join", payload, self.timeout)
        return str(resp["client_id"])

    def heartbeat(self, client_id: str) -> Dict[str, Any]:
        payload = {"client_id": client_id, "timestamp": time.time()}
        return _request_json("POST", f"{self.base_url}/api/v1/heartbeat", payload, self.timeout)

    def get_model(self, client_id: str) -> Dict[str, Any]:
        payload = {"client_id": client_id}
        return _request_json("POST", f"{self.base_url}/api/v1/get_model", payload, self.timeout)

    def submit_update(
        self,
        client_id: str,
        round_id: int,
        num_samples: int,
        train_loss: float,
        model_state: Optional[Dict[str, Any]] = None,
        delta_state: Optional[Dict[str, Any]] = None,
        compressed_update: Optional[Dict[str, Any]] = None,
        compressed_type: Optional[str] = None,
        epsilon: Optional[float] = None,
        cosine_score: Optional[float] = None,
        masked_update: Optional[Dict[str, Any]] = None,
        update_norm: Optional[float] = None,
        train_accuracy: Optional[float] = None,
        upload_bytes: Optional[int] = None,
        download_bytes: Optional[int] = None,
        raw_bytes: Optional[int] = None,
        compressed_bytes: Optional[int] = None,
        train_time_ms: Optional[float] = None,
        label_histogram: Optional[Dict[str, int]] = None,
        train_steps: Optional[int] = None,
        pre_dp_norm: Optional[float] = None,
        clip_applied: Optional[bool] = None,
        attack_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "client_id": client_id,
            "round_id": round_id,
            "num_samples": num_samples,
            "train_loss": train_loss,
        }
        if model_state is not None:
            payload["model_state"] = model_state
        if delta_state is not None:
            payload["delta_state"] = delta_state
        if compressed_update is not None:
            payload["compressed_update"] = compressed_update
        if compressed_type is not None:
            payload["compressed_type"] = compressed_type
        if epsilon is not None:
            payload["epsilon"] = epsilon
        if cosine_score is not None:
            payload["cosine_score"] = cosine_score
        if masked_update is not None:
            payload["masked_update"] = masked_update
        if update_norm is not None:
            payload["update_norm"] = update_norm
        if train_accuracy is not None:
            payload["train_accuracy"] = train_accuracy
        if upload_bytes is not None:
            payload["upload_bytes"] = upload_bytes
        if download_bytes is not None:
            payload["download_bytes"] = download_bytes
        if raw_bytes is not None:
            payload["raw_bytes"] = raw_bytes
        if compressed_bytes is not None:
            payload["compressed_bytes"] = compressed_bytes
        if train_time_ms is not None:
            payload["train_time_ms"] = train_time_ms
        if label_histogram is not None:
            payload["label_histogram"] = label_histogram
        if train_steps is not None:
            payload["train_steps"] = train_steps
        if pre_dp_norm is not None:
            payload["pre_dp_norm"] = pre_dp_norm
        if clip_applied is not None:
            payload["clip_applied"] = clip_applied
        if attack_type is not None:
            payload["attack_type"] = attack_type
        return _request_json("POST", f"{self.base_url}/api/v1/submit_update", payload, self.timeout)
