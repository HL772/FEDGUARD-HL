import math
from typing import Dict, List, Optional, Tuple

import torch


def _state_norm(delta_state: Dict[str, list]) -> float:
    total = 0.0
    for value in delta_state.values():
        tensor = torch.tensor(value, dtype=torch.float32)
        total += float(torch.sum(tensor ** 2).item())
    return math.sqrt(total)


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 0:
        return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])
    return sorted_vals[mid]


def _mad(values: List[float], median: float) -> float:
    if not values:
        return 0.0
    deviations = [abs(v - median) for v in values]
    return _median(deviations)


def _flatten_state(state: Dict[str, list], keys: List[str]) -> torch.Tensor:
    if not keys:
        return torch.zeros(1, dtype=torch.float32)
    tensors = [torch.tensor(state[key], dtype=torch.float32).reshape(-1) for key in keys]
    return torch.cat(tensors)


def _cosine_similarity(vec: torch.Tensor, mean_vec: torch.Tensor) -> float:
    denom = torch.norm(vec) * torch.norm(mean_vec)
    if denom.item() == 0.0:
        return 0.0
    return float(torch.dot(vec, mean_vec) / denom)


class MaliciousDetectionAgent:
    def __init__(
        self,
        loss_threshold: float = 3.0,
        norm_threshold: float = 3.0,
        require_both: bool = True,
        min_mad: float = 0.05,
        cosine_threshold: float = 2.5,
        cosine_enabled: bool = False,
        cosine_top_k: int = 5,
    ) -> None:
        self.loss_threshold = loss_threshold
        self.norm_threshold = norm_threshold
        self.require_both = require_both
        self.min_mad = min_mad
        self.cosine_threshold = cosine_threshold
        self.cosine_enabled = cosine_enabled
        self.cosine_top_k = cosine_top_k

    def detect(self, updates: List[Dict[str, object]]) -> Dict[str, object]:
        losses: List[float] = []
        norms: List[float] = []
        reported_scores: Dict[str, float] = {}
        for update in updates:
            losses.append(float(update.get("train_loss", 0.0)))
            if update.get("pre_dp_norm") is not None:
                norms.append(float(update["pre_dp_norm"]))
            elif "update_norm" in update and update["update_norm"] is not None:
                norms.append(float(update["update_norm"]))
            elif "delta_state" in update:
                norms.append(_state_norm(update["delta_state"]))
            elif "model_state" in update:
                norms.append(_state_norm(update["model_state"]))
            else:
                norms.append(0.0)
            if self.cosine_enabled:
                score = update.get("cosine_score")
                if score is not None:
                    reported_scores[str(update.get("client_id", ""))] = float(score)

        loss_median = _median(losses)
        loss_mad = max(_mad(losses, loss_median), self.min_mad)
        norm_median = _median(norms)
        norm_mad = max(_mad(norms, norm_median), self.min_mad)

        cosine_scores: Dict[str, float] = {}
        cosine_flags: Dict[str, bool] = {}
        similarity_rank: List[Tuple[str, float]] = []
        cosine_threshold = self.cosine_threshold
        cos_median = 0.0
        cos_mad = self.min_mad
        if self.cosine_enabled:
            vectors: List[torch.Tensor] = []
            vector_clients: List[str] = []
            keys: Optional[List[str]] = None
            for update in updates:
                state = update.get("delta_state") or update.get("model_state")
                if state is None:
                    continue
                state_keys = sorted(state.keys())
                if keys is None:
                    keys = state_keys
                if state_keys != keys:
                    continue
                vectors.append(_flatten_state(state, keys))
                vector_clients.append(str(update.get("client_id", "")))
            if len(vectors) >= 2:
                stacked = torch.stack(vectors, dim=0)
                mean_vec = torch.mean(stacked, dim=0)
                for client_id, vec in zip(vector_clients, vectors):
                    cosine_scores[client_id] = _cosine_similarity(vec, mean_vec)
                cos_values = list(cosine_scores.values())
                cos_median = _median(cos_values)
                cos_mad = max(_mad(cos_values, cos_median), self.min_mad)
                for client_id, cos_val in cosine_scores.items():
                    cos_score = (cos_median - cos_val) / (cos_mad + 1e-6)
                    cosine_flags[client_id] = cos_score > cosine_threshold
                limit = self.cosine_top_k if self.cosine_top_k > 0 else len(cosine_scores)
                similarity_rank = sorted(
                    cosine_scores.items(), key=lambda item: item[1]
                )[:limit]
            elif reported_scores:
                cosine_scores = reported_scores
                cos_values = list(cosine_scores.values())
                cos_median = _median(cos_values)
                cos_mad = max(_mad(cos_values, cos_median), self.min_mad)
                for client_id, cos_val in cosine_scores.items():
                    cos_score = (cos_median - cos_val) / (cos_mad + 1e-6)
                    cosine_flags[client_id] = cos_score > cosine_threshold
                limit = self.cosine_top_k if self.cosine_top_k > 0 else len(cosine_scores)
                similarity_rank = sorted(
                    cosine_scores.items(), key=lambda item: item[1]
                )[:limit]

        excluded: List[str] = []
        anomaly_scores: Dict[str, float] = {}
        excluded_reasons: Dict[str, List[str]] = {}
        for idx, update in enumerate(updates):
            loss = losses[idx]
            norm = norms[idx]
            loss_score = (loss - loss_median) / (loss_mad + 1e-6)
            norm_score = (norm - norm_median) / (norm_mad + 1e-6)
            client_id = str(update.get("client_id", idx))
            cos_flag = cosine_flags.get(client_id, False)
            cos_score = 0.0
            if client_id in cosine_scores:
                cos_score = (cos_median - cosine_scores[client_id]) / (cos_mad + 1e-6)
            score = max(loss_score, norm_score, cos_score)
            anomaly_scores[client_id] = float(score)
            loss_flag = loss_score > self.loss_threshold
            norm_flag = norm_score > self.norm_threshold
            if self.cosine_enabled:
                if self.require_both:
                    is_anom = (
                        (loss_flag and norm_flag)
                        or (loss_flag and cos_flag)
                        or (norm_flag and cos_flag)
                    )
                else:
                    is_anom = loss_flag or norm_flag or cos_flag
            else:
                if self.require_both:
                    is_anom = loss_flag and norm_flag
                else:
                    is_anom = loss_flag or norm_flag
            if is_anom:
                excluded.append(client_id)
                reasons: List[str] = []
                if loss_flag:
                    reasons.append("loss")
                if norm_flag:
                    reasons.append("norm")
                if cos_flag:
                    reasons.append("cosine")
                excluded_reasons[client_id] = reasons

        return {
            "excluded_clients": excluded,
            "anomaly_scores": anomaly_scores,
            "cosine_scores": cosine_scores,
            "similarity_rank": similarity_rank,
            "excluded_reasons": excluded_reasons,
            "cosine_threshold": cosine_threshold if self.cosine_enabled else 0.0,
        }
