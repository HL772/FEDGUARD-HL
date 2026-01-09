from typing import Dict, List, Tuple

import torch


def median_aggregate(states: List[Dict[str, list]]) -> Dict[str, list]:
    if not states:
        raise ValueError("No states to aggregate")
    result: Dict[str, list] = {}
    keys = states[0].keys()
    for key in keys:
        tensors = torch.stack([torch.tensor(state[key], dtype=torch.float32) for state in states], dim=0)
        median = torch.median(tensors, dim=0).values
        result[key] = median.cpu().tolist()
    return result


def trimmed_mean_aggregate(states: List[Dict[str, list]], trim_ratio: float) -> Dict[str, list]:
    if not states:
        raise ValueError("No states to aggregate")
    if trim_ratio < 0 or trim_ratio >= 0.5:
        raise ValueError("trim_ratio must be in [0, 0.5)")
    n = len(states)
    trim_k = int(n * trim_ratio)
    result: Dict[str, list] = {}
    keys = states[0].keys()
    for key in keys:
        tensors = torch.stack([torch.tensor(state[key], dtype=torch.float32) for state in states], dim=0)
        if trim_k > 0 and n - 2 * trim_k > 0:
            sorted_vals, _ = torch.sort(tensors, dim=0)
            trimmed = sorted_vals[trim_k : n - trim_k]
            mean = torch.mean(trimmed, dim=0)
        else:
            mean = torch.mean(tensors, dim=0)
        result[key] = mean.cpu().tolist()
    return result


def _flatten_state(state: Dict[str, list], keys: List[str]) -> torch.Tensor:
    tensors = [torch.tensor(state[key], dtype=torch.float32).reshape(-1) for key in keys]
    if not tensors:
        return torch.zeros(1, dtype=torch.float32)
    return torch.cat(tensors)


def _pairwise_distances(vectors: List[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(vectors, dim=0)
    return torch.cdist(stacked, stacked, p=2)


def krum_aggregate(states: List[Dict[str, list]], byzantine_f: int = 1) -> Dict[str, list]:
    if not states:
        raise ValueError("No states to aggregate")
    n = len(states)
    f = max(int(byzantine_f), 0)
    if n <= 2 * f + 2:
        return trimmed_mean_aggregate(states, 0.0)
    keys = sorted(states[0].keys())
    vectors = [_flatten_state(state, keys) for state in states]
    distances = _pairwise_distances(vectors)
    scores: List[Tuple[int, float]] = []
    k = max(n - f - 2, 1)
    for idx in range(n):
        sorted_vals, _ = torch.sort(distances[idx])
        score = float(torch.sum(sorted_vals[1 : 1 + k]).item())
        scores.append((idx, score))
    selected_idx = min(scores, key=lambda item: item[1])[0]
    return states[selected_idx]


def bulyan_aggregate(states: List[Dict[str, list]], byzantine_f: int = 1) -> Dict[str, list]:
    if not states:
        raise ValueError("No states to aggregate")
    n = len(states)
    f = max(int(byzantine_f), 0)
    if n <= 4 * f + 3:
        return krum_aggregate(states, byzantine_f=f)
    candidate_count = max(n - 2 * f, 1)
    candidates = list(states)
    selected: List[Dict[str, list]] = []
    while candidates and len(selected) < candidate_count:
        chosen = krum_aggregate(candidates, byzantine_f=f)
        selected.append(chosen)
        candidates.remove(chosen)
    if not selected:
        return krum_aggregate(states, byzantine_f=f)
    keys = selected[0].keys()
    result: Dict[str, list] = {}
    for key in keys:
        tensors = torch.stack(
            [torch.tensor(state[key], dtype=torch.float32) for state in selected], dim=0
        )
        sorted_vals, _ = torch.sort(tensors, dim=0)
        trim_k = min(f, sorted_vals.size(0) // 2)
        if trim_k > 0 and sorted_vals.size(0) - 2 * trim_k > 0:
            trimmed = sorted_vals[trim_k : sorted_vals.size(0) - trim_k]
        else:
            trimmed = sorted_vals
        mean = torch.mean(trimmed, dim=0)
        result[key] = mean.cpu().tolist()
    return result
