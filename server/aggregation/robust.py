from typing import Dict, List, Tuple

import torch

# 鲁棒聚合工具：用于抵抗异常/恶意更新


def median_aggregate(states: List[Dict[str, list]]) -> Dict[str, list]:
    # 坐标维度逐点取中位数（逐维鲁棒聚合）
    if not states:
        raise ValueError("No states to aggregate")
    result: Dict[str, list] = {}
    keys = states[0].keys()  # 假设所有客户端参数键一致
    for key in keys:
        # 将同一参数在不同客户端的值堆叠成 [num_clients, *shape]
        tensors = torch.stack([torch.tensor(state[key], dtype=torch.float32) for state in states], dim=0)
        # 逐维中位数，抵抗离群值
        median = torch.median(tensors, dim=0).values
        result[key] = median.cpu().tolist()
    return result


def trimmed_mean_aggregate(states: List[Dict[str, list]], trim_ratio: float) -> Dict[str, list]:
    # 对每个坐标去掉极值后取均值（trimmed mean）
    if not states:
        raise ValueError("No states to aggregate")
    if trim_ratio < 0 or trim_ratio >= 0.5:
        raise ValueError("trim_ratio must be in [0, 0.5)")
    n = len(states)
    trim_k = int(n * trim_ratio)  # 两端各剔除的客户端数量
    result: Dict[str, list] = {}
    keys = states[0].keys()
    for key in keys:
        # 堆叠成 [num_clients, *shape] 方便逐维排序
        tensors = torch.stack([torch.tensor(state[key], dtype=torch.float32) for state in states], dim=0)
        if trim_k > 0 and n - 2 * trim_k > 0:
            # 逐维排序并移除两端极值
            sorted_vals, _ = torch.sort(tensors, dim=0)
            trimmed = sorted_vals[trim_k : n - trim_k]
            # 剩余取均值
            mean = torch.mean(trimmed, dim=0)
        else:
            # 剔除比例为 0 或样本数不足时退化为均值
            mean = torch.mean(tensors, dim=0)
        result[key] = mean.cpu().tolist()
    return result


def _flatten_state(state: Dict[str, list], keys: List[str]) -> torch.Tensor:
    # 展平成向量用于距离评分（Krum / Bulyan）
    tensors = [torch.tensor(state[key], dtype=torch.float32).reshape(-1) for key in keys]
    if not tensors:
        return torch.zeros(1, dtype=torch.float32)
    return torch.cat(tensors)


def _pairwise_distances(vectors: List[torch.Tensor]) -> torch.Tensor:
    # 计算两两欧氏距离矩阵（Krum 选择使用）
    stacked = torch.stack(vectors, dim=0)
    return torch.cdist(stacked, stacked, p=2)


def krum_aggregate(states: List[Dict[str, list]], byzantine_f: int = 1) -> Dict[str, list]:
    # Krum：选择与其他更新最接近的一个更新
    if not states:
        raise ValueError("No states to aggregate")
    n = len(states)
    f = max(int(byzantine_f), 0)  # 假设的最大恶意客户端数
    if n <= 2 * f + 2:
        # 样本不足时退化为普通均值
        return trimmed_mean_aggregate(states, 0.0)
    keys = sorted(states[0].keys())  # 对齐参数顺序
    vectors = [_flatten_state(state, keys) for state in states]  # 每个客户端更新向量化
    distances = _pairwise_distances(vectors)  # 两两距离矩阵
    scores: List[Tuple[int, float]] = []
    k = max(n - f - 2, 1)  # 每个候选只看最近的 k 个距离
    for idx in range(n):
        sorted_vals, _ = torch.sort(distances[idx])
        # 跳过自身距离（第一个是 0），累加最近 k 个邻居距离
        score = float(torch.sum(sorted_vals[1 : 1 + k]).item())
        scores.append((idx, score))
    # 选择距离和最小的客户端更新
    selected_idx = min(scores, key=lambda item: item[1])[0]
    return states[selected_idx]


def bulyan_aggregate(states: List[Dict[str, list]], byzantine_f: int = 1) -> Dict[str, list]:
    # Bulyan：迭代 Krum 选出候选，再做 trimmed mean
    if not states:
        raise ValueError("No states to aggregate")
    n = len(states)
    f = max(int(byzantine_f), 0)  # 假设的最大恶意客户端数
    if n <= 4 * f + 3:
        # 样本不足时退化为 Krum
        return krum_aggregate(states, byzantine_f=f)
    candidate_count = max(n - 2 * f, 1)  # 候选集合规模
    candidates = list(states)
    selected: List[Dict[str, list]] = []
    while candidates and len(selected) < candidate_count:
        # 反复用 Krum 选出可信候选
        chosen = krum_aggregate(candidates, byzantine_f=f)
        selected.append(chosen)
        candidates.remove(chosen)
    if not selected:
        return krum_aggregate(states, byzantine_f=f)
    keys = selected[0].keys()
    result: Dict[str, list] = {}
    for key in keys:
        # 在候选集上做逐维 trimmed mean
        tensors = torch.stack(
            [torch.tensor(state[key], dtype=torch.float32) for state in selected], dim=0
        )
        sorted_vals, _ = torch.sort(tensors, dim=0)
        trim_k = min(f, sorted_vals.size(0) // 2)  # 防止剔除过多导致空集
        if trim_k > 0 and sorted_vals.size(0) - 2 * trim_k > 0:
            trimmed = sorted_vals[trim_k : sorted_vals.size(0) - trim_k]
        else:
            trimmed = sorted_vals
        mean = torch.mean(trimmed, dim=0)
        result[key] = mean.cpu().tolist()
    return result
