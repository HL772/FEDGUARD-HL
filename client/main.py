import argparse
import json
import os
import subprocess
import sys
import time
from fnmatch import fnmatch
from typing import Any, Dict, Iterable, List, Optional, Set

import torch

from client.comm.api_client import ApiClient, ApiError
from client.compression.error_feedback import ErrorFeedbackAgent
from client.compression.topk import CompressionAgent
from client.data.partition import get_client_loader
from client.privacy.dp import apply_dp
from client.secure.mask import SecureMaskingAgent
from client.security.attack import AttackSimulator
from client.train.local_trainer import (
    create_model,
    extract_state_by_keys,
    load_state_dict_from_list,
    state_dict_to_list,
    train_one_epoch,
    train_one_epoch_dual,
    update_state_dict_from_list,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedGuard client process")
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8000",
        help="Server base URL",
    )
    parser.add_argument("--client-name", default=None, help="Client display name")
    parser.add_argument("--heartbeat-interval", type=float, default=1.0)
    parser.add_argument("--max-heartbeats", type=int, default=3)
    parser.add_argument("--join-timeout", type=float, default=10.0)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    parser.add_argument("--client-rank", type=int, default=0)
    parser.add_argument("--num-clients", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--config", default="experiments/configs/default.yaml")
    parser.add_argument("--single-round", action="store_true")
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--heartbeat-only", action="store_true")
    parser.add_argument(
        "--stay-online-on-not-selected",
        action="store_true",
        help="Continue sending heartbeats when not selected",
    )
    return parser.parse_args()


def _register_with_retry(api: ApiClient, client_name: Optional[str], join_timeout: float) -> str:
    deadline = time.time() + join_timeout
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            return api.join(client_name=client_name)
        except ApiError as exc:
            print(f"[client] join attempt {attempt} failed: {exc}", flush=True)
            time.sleep(0.5)
    raise ApiError("Join timed out")


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


def _normalize_pattern_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


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


def _resolve_split_keys(
    keys: Iterable[str], backbone_patterns: List[str], head_patterns: List[str]
) -> tuple[Set[str], Set[str]]:
    key_list = list(keys)
    if not backbone_patterns and not head_patterns:
        return set(key_list), set()
    backbone = _match_patterns(key_list, backbone_patterns)
    head = _match_patterns(key_list, head_patterns)
    if backbone and not head_patterns:
        head = set(key_list) - backbone
    if head and not backbone_patterns:
        backbone = set(key_list) - head
    if not backbone:
        backbone = set(key_list)
    return backbone, head


def _cosine_similarity(
    left: Dict[str, list], right: Dict[str, list]
) -> Optional[float]:
    if not left or not right:
        return None
    common = sorted(set(left.keys()) & set(right.keys()))
    if not common:
        return None
    left_vec = torch.cat(
        [torch.tensor(left[key], dtype=torch.float32).reshape(-1) for key in common]
    )
    right_vec = torch.cat(
        [torch.tensor(right[key], dtype=torch.float32).reshape(-1) for key in common]
    )
    denom = torch.norm(left_vec) * torch.norm(right_vec)
    if denom.item() == 0.0:
        return 0.0
    return float(torch.dot(left_vec, right_vec) / denom)


def _resolve_private_keys(keys: Iterable[str], private_patterns: List[str]) -> Set[str]:
    return _match_patterns(keys, private_patterns)


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


def _heartbeat_lock_path(client_name: str) -> str:
    state_dir = os.environ.get("CLIENT_HEARTBEAT_STATE_DIR", "experiments/runtime/heartbeats")
    os.makedirs(state_dir, exist_ok=True)
    safe_name = client_name.replace(os.sep, "_")
    return os.path.join(state_dir, f"{safe_name}.pid")


def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _ensure_heartbeat_process(args: argparse.Namespace) -> None:
    if not args.client_name:
        return
    lock_path = _heartbeat_lock_path(args.client_name)
    if os.path.exists(lock_path):
        try:
            with open(lock_path, "r", encoding="utf-8") as handle:
                existing_pid = int(handle.read().strip() or "0")
            if existing_pid > 0 and _is_process_alive(existing_pid):
                return
        except Exception:
            pass
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "client.main",
        "--client-name",
        args.client_name,
        "--heartbeat-only",
        "--server-url",
        args.server_url,
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--join-timeout",
        str(args.join_timeout),
        "--request-timeout",
        str(args.request_timeout),
        "--config",
        args.config,
    ]
    if args.download_data:
        cmd.append("--download-data")
    subprocess.Popen(cmd, cwd=os.getcwd(), env=os.environ.copy())


def main() -> int:
    args = _parse_args()
    api = ApiClient(args.server_url, timeout=args.request_timeout)
    try:
        client_id = _register_with_retry(api, args.client_name, args.join_timeout)
    except ApiError as exc:
        print(f"[client] failed to join server: {exc}", flush=True)
        return 1

    print(f"[client] registered client_id={client_id}", flush=True)
    if args.register_only:
        try:
            api.heartbeat(client_id)
        except ApiError as exc:
            print(f"[client] heartbeat failed: {exc}", flush=True)
        return 0
    if args.heartbeat_only:
        if args.client_name:
            lock_path = _heartbeat_lock_path(args.client_name)
            try:
                with open(lock_path, "w", encoding="utf-8") as handle:
                    handle.write(str(os.getpid()))
            except Exception:
                pass
        while True:
            try:
                api.heartbeat(client_id)
            except ApiError as exc:
                print(f"[client] heartbeat failed: {exc}", flush=True)
            time.sleep(args.heartbeat_interval)
    try:
        api.heartbeat(client_id)
    except ApiError as exc:
        print(f"[client] heartbeat failed: {exc}", flush=True)

    config = _load_config(args.config)
    dp_config = config.get("dp", {})
    compression_config = config.get("compression", {})
    security_config = config.get("security", {})
    train_config = config.get("train", {})
    model_config = config.get("model", {})
    split_config = model_config.get("split", {})

    dp_enabled = bool(dp_config.get("enabled", False))
    clip_norm = float(dp_config.get("clip_norm", 1.0))
    noise_multiplier = float(dp_config.get("noise_multiplier", 0.0))
    dp_delta = float(dp_config.get("delta", 1e-5))
    train_algo = str(train_config.get("algo", "fedavg")).lower()
    backbone_patterns = _normalize_pattern_list(split_config.get("backbone_keys"))
    head_patterns = _normalize_pattern_list(split_config.get("head_keys"))
    private_head_patterns = _normalize_pattern_list(split_config.get("private_head_keys"))
    private_head_lr = train_config.get("private_head_lr")
    if private_head_lr is not None:
        private_head_lr = float(private_head_lr)
    private_head_epochs = int(train_config.get("private_head_epochs", 1))

    compression_enabled = bool(compression_config.get("enabled", False))
    topk_ratio = float(compression_config.get("topk_ratio", 1.0))
    quant_bits = int(compression_config.get("quant_bits", 8))
    error_feedback_enabled = bool(compression_config.get("error_feedback", False))
    compressor = CompressionAgent(topk_ratio=topk_ratio, quant_bits=quant_bits)
    secure_compressor = CompressionAgent(topk_ratio=1.0, quant_bits=quant_bits)
    error_feedback = ErrorFeedbackAgent()

    secure_aggregation = bool(security_config.get("secure_aggregation", False))
    mask_scale = float(security_config.get("mask_scale", 1.0))
    masking_agent = SecureMaskingAgent(mask_scale=mask_scale)
    attack_cfg = security_config.get("attack_simulation", {})
    attack_enabled = bool(attack_cfg.get("enabled", False))
    attack_method = str(attack_cfg.get("method", "none")).lower()
    attack_scale = float(attack_cfg.get("scale", 1.0))
    attack_label_flip = bool(attack_cfg.get("label_flip", False)) or attack_method == "label_flip"
    attack_loss_scale = float(attack_cfg.get("loss_scale", 1.0))
    attack_accuracy_scale = float(attack_cfg.get("accuracy_scale", 1.0))
    attack_malicious_ranks = attack_cfg.get("malicious_ranks") or []
    attack_malicious_fraction = float(attack_cfg.get("malicious_fraction", 0.0))
    attack_simulator = AttackSimulator(method=attack_method, scale=attack_scale)
    is_malicious = False
    if attack_enabled:
        is_malicious = attack_simulator.is_malicious(
            args.client_rank,
            args.num_clients,
            malicious_ranks=attack_malicious_ranks,
            malicious_fraction=attack_malicious_fraction,
        )

    try:
        data_config = config.get("data", {})
        alpha = float(data_config.get("alpha", args.alpha))
        train_loader, label_hist = get_client_loader(
            client_rank=args.client_rank,
            num_clients=args.num_clients,
            alpha=alpha,
            seed=42,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            download=args.download_data,
        )
    except Exception as exc:
        print(f"[client] data load failed: {exc}", flush=True)
        return 1
    print(f"[client] label_histogram={label_hist}", flush=True)
    label_hist_payload = {str(label): int(count) for label, count in label_hist.items()}

    device = torch.device("cpu")
    last_heartbeat = 0.0
    last_wait_reason = ""
    last_wait_log = 0.0
    private_head_state: Optional[Dict[str, list]] = None
    while True:
        if time.time() - last_heartbeat >= args.heartbeat_interval:
            try:
                api.heartbeat(client_id)
            except ApiError as exc:
                print(f"[client] heartbeat failed: {exc}", flush=True)
            last_heartbeat = time.time()
        try:
            resp = api.get_model(client_id)
        except ApiError as exc:
            print(f"[client] get_model failed: {exc}", flush=True)
            time.sleep(args.heartbeat_interval)
            continue

        status = resp.get("status")
        if status == "wait":
            wait_reason = str(resp.get("reason") or "")
            now = time.time()
            if wait_reason != last_wait_reason or now - last_wait_log >= 10.0:
                print(f"[client] wait reason={wait_reason}", flush=True)
                last_wait_reason = wait_reason
                last_wait_log = now
            if args.single_round and wait_reason in ("not_selected", "timeout"):
                print(f"[client] skipped round ({wait_reason})", flush=True)
                if args.stay_online_on_not_selected:
                    _ensure_heartbeat_process(args)
                break
            time.sleep(args.heartbeat_interval)
            continue
        if status == "done":
            print("[client] training complete", flush=True)
            break

        round_id = int(resp["round_id"])
        model_state = resp["model_state"]
        round_config = resp.get("config", {})
        participants = resp.get("participants", [])
        lr = float(round_config.get("lr", 0.1))
        mu = float(round_config.get("mu", 0.0))
        round_algo = str(round_config.get("algo", train_algo)).lower()
        round_secure = bool(round_config.get("secure_aggregation", secure_aggregation))
        round_dp = round_config.get("dp_params", {})
        cosine_ref = round_config.get("cosine_ref")
        dp_enabled_round = bool(round_dp.get("enabled", dp_enabled))
        clip_norm_round = float(round_dp.get("clip_norm", clip_norm))
        noise_multiplier_round = float(round_dp.get("noise_multiplier", noise_multiplier))
        dp_delta_round = float(round_dp.get("delta", dp_delta))

        model = create_model()
        load_state_dict_from_list(model, model_state)
        backbone_keys: Set[str] = set()
        head_keys: Set[str] = set()
        private_keys: Set[str] = set()
        aggregated_keys: Set[str] = set()
        if round_algo in ("fedper", "fedper_dual"):
            if not backbone_patterns and not head_patterns:
                backbone_patterns = ["backbone.*"]
                head_patterns = ["head.*"]
            backbone_keys, head_keys = _resolve_split_keys(
                model_state.keys(), backbone_patterns, head_patterns
            )
            if round_algo == "fedper_dual":
                if not private_head_patterns:
                    private_head_patterns = ["private_head.*"]
                private_keys = _resolve_private_keys(model_state.keys(), private_head_patterns)
                aggregated_keys = backbone_keys | head_keys
            else:
                private_keys = head_keys
                aggregated_keys = backbone_keys
            if private_keys:
                if private_head_state is None:
                    private_head_state = {
                        key: model_state[key] for key in private_keys if key in model_state
                    }
                else:
                    update_state_dict_from_list(model, private_head_state)
        start_train = time.time()
        if round_algo == "fedper_dual":
            shared_params = {}
            for name, param in model.named_parameters():
                if name in aggregated_keys:
                    shared_params[name] = param.detach().clone()
            loss, num_samples, steps, accuracy = train_one_epoch_dual(
                model,
                train_loader,
                lr=lr,
                device=device,
                mu=mu,
                global_params=shared_params,
                private_lr=private_head_lr,
                private_epochs=private_head_epochs,
                label_flip=attack_label_flip and is_malicious,
            )
        else:
            global_params = [param.detach().clone() for param in model.parameters()]
            loss, num_samples, steps, accuracy = train_one_epoch(
                model,
                train_loader,
                lr=lr,
                device=device,
                mu=mu,
                global_params=global_params,
                label_flip=attack_label_flip and is_malicious,
            )
        train_time_ms = (time.time() - start_train) * 1000.0
        reported_loss = loss
        reported_accuracy = accuracy
        if attack_enabled and is_malicious:
            reported_loss = loss * attack_loss_scale
            reported_accuracy = max(0.0, min(1.0, accuracy * attack_accuracy_scale))
        if round_algo in ("fedper", "fedper_dual") and private_keys:
            private_head_state = extract_state_by_keys(model, private_keys)
        updated_state = state_dict_to_list(model)

        delta_state: Dict[str, list] = {}
        for key, value in updated_state.items():
            base_tensor = torch.tensor(model_state[key], dtype=torch.float32)
            delta_tensor = torch.tensor(value, dtype=torch.float32) - base_tensor
            delta_state[key] = delta_tensor.cpu().tolist()
        if aggregated_keys:
            delta_state = {key: value for key, value in delta_state.items() if key in aggregated_keys}

        attack_type = "none"
        if attack_enabled and is_malicious:
            attack_type = attack_method
            if attack_method in ("sign_flip", "scale"):
                delta_state = attack_simulator.apply(delta_state)

        cosine_score = None
        if isinstance(cosine_ref, dict):
            cosine_score = _cosine_similarity(delta_state, cosine_ref)

        epsilon = 0.0
        pre_dp_norm = None
        clip_applied = None
        if dp_enabled_round:
            sample_rate = min(1.0, float(args.batch_size) / max(num_samples, 1))
            delta_state, epsilon, pre_dp_norm, clip_applied = apply_dp(
                delta_state,
                clip_norm=clip_norm_round,
                noise_multiplier=noise_multiplier_round,
                delta=dp_delta_round,
                sample_rate=sample_rate,
                steps=max(steps, 1),
            )

        update_norm = 0.0
        for value in delta_state.values():
            tensor = torch.tensor(value, dtype=torch.float32)
            update_norm += float(torch.sum(tensor ** 2).item())
        update_norm = float(update_norm ** 0.5)

        masked_update = None
        compressed_update = None
        compressed_type = None
        if round_secure:
            if not participants:
                print("[client] missing participants list for secure aggregation", flush=True)
                time.sleep(args.heartbeat_interval)
                continue
            scaled_delta: Dict[str, list] = {}
            for key, value in delta_state.items():
                tensor = torch.tensor(value, dtype=torch.float32) * float(num_samples)
                scaled_delta[key] = tensor.cpu().tolist()
            masked_update = masking_agent.apply_mask(
                scaled_delta,
                participants=participants,
                client_id=client_id,
                round_id=round_id,
            )
            if compression_enabled:
                compressed_update = secure_compressor.compress(masked_update)
                compressed_type = "masked"
                masked_update = None
        else:
            if compression_enabled:
                ef_state = delta_state
                if error_feedback_enabled:
                    ef_state = error_feedback.apply(delta_state)
                compressed_update = compressor.compress(ef_state)
                if error_feedback_enabled:
                    reconstructed = compressor.decompress(compressed_update)
                    error_feedback.update(ef_state, reconstructed)
                compressed_type = "delta"

        try:
            delta_payload = None if round_secure or compression_enabled else delta_state
            payload_obj = masked_update or compressed_update or delta_payload or {}
            upload_bytes = len(json.dumps(payload_obj).encode("utf-8"))
            download_bytes = len(json.dumps(model_state).encode("utf-8"))
            raw_bytes = len(json.dumps(delta_state).encode("utf-8"))
            if compressed_update is not None:
                compressed_bytes = len(json.dumps(compressed_update).encode("utf-8"))
            else:
                compressed_bytes = upload_bytes
            submit_resp = api.submit_update(
                client_id,
                round_id,
                num_samples,
                reported_loss,
                delta_state=delta_payload,
                compressed_update=compressed_update,
                compressed_type=compressed_type,
                epsilon=epsilon if dp_enabled else None,
                cosine_score=cosine_score,
                masked_update=masked_update,
                update_norm=update_norm,
                train_accuracy=reported_accuracy,
                upload_bytes=upload_bytes,
                download_bytes=download_bytes,
                raw_bytes=raw_bytes,
                compressed_bytes=compressed_bytes,
                train_time_ms=train_time_ms,
                label_histogram=label_hist_payload,
                train_steps=steps,
                pre_dp_norm=pre_dp_norm,
                clip_applied=clip_applied,
                attack_type=attack_type,
            )
            print(
                f"[client] round {round_id} submitted loss={reported_loss:.4f} acc={reported_accuracy:.4f} epsilon={epsilon:.4f} resp={submit_resp}",
                flush=True,
            )
            if args.single_round:
                break
        except ApiError as exc:
            print(f"[client] submit_update failed: {exc}", flush=True)
            time.sleep(args.heartbeat_interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
