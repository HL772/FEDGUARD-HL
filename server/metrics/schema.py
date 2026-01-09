from typing import Any, Dict, List


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_round_metric(metric: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(metric)
    data["round_id"] = _as_int(data.get("round_id"), 0)
    data["global_loss"] = _as_float(data.get("global_loss"), 0.0)
    data["global_accuracy"] = _as_float(data.get("global_accuracy"), 0.0)
    data["participants"] = list(data.get("participants") or [])
    data["dropped_clients"] = list(data.get("dropped_clients") or [])
    data["alerts"] = list(data.get("alerts") or [])
    data["online_clients"] = list(data.get("online_clients") or [])
    data["client_updates"] = list(data.get("client_updates") or [])
    client_counts = data.get("client_counts") or {}
    data["client_counts"] = {
        "online": _as_int(client_counts.get("online"), 0),
        "eligible": _as_int(client_counts.get("eligible"), 0),
        "registered": _as_int(client_counts.get("registered"), 0),
        "blacklisted": _as_int(client_counts.get("blacklisted"), 0),
    }

    privacy = dict(data.get("privacy") or {})
    privacy["enabled"] = bool(privacy.get("enabled", False))
    privacy["mode"] = str(privacy.get("mode") or "off")
    privacy["epsilon_accountant"] = _as_float(privacy.get("epsilon_accountant"), 0.0)
    privacy["target_epsilon"] = _as_float(privacy.get("target_epsilon"), 0.0)
    privacy["epsilon_max"] = _as_float(privacy.get("epsilon_max"), 0.0)
    privacy["epsilon_avg"] = _as_float(privacy.get("epsilon_avg"), 0.0)
    privacy["delta"] = _as_float(privacy.get("delta"), 0.0)
    privacy["noise_multiplier"] = _as_float(privacy.get("noise_multiplier"), 0.0)
    privacy["clip_norm"] = _as_float(privacy.get("clip_norm"), 0.0)
    privacy["clip_rate"] = _as_float(privacy.get("clip_rate"), 0.0)
    data["privacy"] = privacy

    comm = dict(data.get("comm") or {})
    comm["upload_bytes_total"] = _as_int(comm.get("upload_bytes_total"), 0)
    comm["download_bytes_total"] = _as_int(comm.get("download_bytes_total"), 0)
    comm["compressed_ratio"] = _as_float(comm.get("compressed_ratio"), 1.0)
    data["comm"] = comm

    robust = dict(data.get("robust") or {})
    robust["agg_method"] = str(robust.get("agg_method") or "fedavg")
    robust["excluded_clients"] = list(robust.get("excluded_clients") or [])
    data["robust"] = robust

    timing = dict(data.get("timing") or {})
    timing["round_wall_time_ms"] = _as_float(timing.get("round_wall_time_ms"), 0.0)
    timing["train_time_ms_avg"] = _as_float(timing.get("train_time_ms_avg"), 0.0)
    data["timing"] = timing

    server = dict(data.get("server") or {})
    server["eval_loss"] = _as_float(server.get("eval_loss"), 0.0)
    server["eval_accuracy"] = _as_float(server.get("eval_accuracy"), 0.0)
    server["update_norm"] = _as_float(server.get("update_norm"), 0.0)
    data["server"] = server

    security = dict(data.get("security") or {})
    attack_sim = dict(security.get("attack_simulation") or {})
    attack_sim["enabled"] = bool(attack_sim.get("enabled", False))
    attack_sim["method"] = str(attack_sim.get("method") or "none")
    attack_sim["malicious_count"] = _as_int(attack_sim.get("malicious_count"), 0)
    security["attack_simulation"] = attack_sim
    detector = dict(security.get("detector") or {})
    detector["precision"] = _as_float(detector.get("precision"), 0.0)
    detector["recall"] = _as_float(detector.get("recall"), 0.0)
    detector["f1"] = _as_float(detector.get("f1"), 0.0)
    detector["cosine_threshold"] = _as_float(detector.get("cosine_threshold"), 0.0)
    detector["true_positive"] = _as_int(detector.get("true_positive"), 0)
    detector["detected"] = _as_int(detector.get("detected"), 0)
    detector["malicious"] = _as_int(detector.get("malicious"), 0)
    security["detector"] = detector
    security["similarity_rank"] = list(security.get("similarity_rank") or [])
    security["cosine_scores"] = dict(security.get("cosine_scores") or {})
    data["security"] = security

    sampling = dict(data.get("sampling") or {})
    sampling["enabled"] = bool(sampling.get("enabled", False))
    sampling["strategy"] = str(sampling.get("strategy") or "random")
    sampling["selection_mode"] = str(sampling.get("selection_mode") or "pre")
    sampling["epsilon"] = _as_float(sampling.get("epsilon"), 0.0)
    sampling["fairness_window"] = _as_int(sampling.get("fairness_window"), 0)
    sampling["timeout_rate"] = _as_float(sampling.get("timeout_rate"), 0.0)
    sampling["selection_histogram"] = dict(sampling.get("selection_histogram") or {})
    sampling["score_rank"] = list(sampling.get("score_rank") or [])
    data["sampling"] = sampling

    fairness = dict(data.get("fairness") or {})
    fairness["client_personal_acc"] = dict(fairness.get("client_personal_acc") or {})
    fairness["avg_accuracy"] = _as_float(fairness.get("avg_accuracy"), 0.0)
    fairness["min_accuracy"] = _as_float(fairness.get("min_accuracy"), 0.0)
    fairness["std_accuracy"] = _as_float(fairness.get("std_accuracy"), 0.0)
    fairness["jain_index"] = _as_float(fairness.get("jain_index"), 0.0)
    data["fairness"] = fairness

    data["deadline_triggered"] = bool(data.get("deadline_triggered", False))
    return data
