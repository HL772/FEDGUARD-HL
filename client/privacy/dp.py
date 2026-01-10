import math
from typing import Dict, Tuple

import torch

# DifferentialPrivacyModule（AGENT.md 3.2.J）：裁剪 + 高斯噪声


def _state_to_tensors(state: Dict[str, list]) -> Dict[str, torch.Tensor]:
    # list -> tensor
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in state.items()}


def _tensors_to_state(tensors: Dict[str, torch.Tensor]) -> Dict[str, list]:
    # tensor -> list
    return {k: v.detach().cpu().tolist() for k, v in tensors.items()}


def _clip_state(
    delta_state: Dict[str, list], clip_norm: float
) -> Tuple[Dict[str, torch.Tensor], float]:
    # L2 裁剪（clip_norm）
    tensors = _state_to_tensors(delta_state)
    total_norm_sq = 0.0
    for tensor in tensors.values():
        total_norm_sq += float(torch.sum(tensor ** 2).item())
    total_norm = math.sqrt(total_norm_sq)

    if clip_norm > 0 and total_norm > clip_norm:
        scale = clip_norm / (total_norm + 1e-12)
        for key in tensors:
            tensors[key] = tensors[key] * scale
    return tensors, total_norm


def apply_dp(
    delta_state: Dict[str, list],
    clip_norm: float,
    noise_multiplier: float,
    delta: float,
    sample_rate: float,
    steps: int,
) -> Tuple[Dict[str, list], float, float, bool]:
    # DP-SGD：裁剪 + 噪声 + ε 估算（简化公式）
    tensors, total_norm = _clip_state(delta_state, clip_norm)
    clipped = clip_norm > 0 and total_norm > clip_norm

    if noise_multiplier > 0:
        std = noise_multiplier * clip_norm
        for key in tensors:
            noise = torch.normal(mean=0.0, std=std, size=tensors[key].shape)
            tensors[key] = tensors[key] + noise

    epsilon = 0.0
    if noise_multiplier > 0 and delta > 0:
        epsilon = (
            sample_rate
            * math.sqrt(2.0 * steps * math.log(1.0 / delta))
            / noise_multiplier
        )

    return _tensors_to_state(tensors), float(epsilon), float(total_norm), clipped
