import math
from typing import Dict, Tuple

import torch

# DifferentialPrivacyModule（AGENT.md 3.2.J）：
# - 客户端侧执行 DP-SGD 的核心步骤（裁剪 + 加噪）
# - 该文件仅负责“执行”，自适应裁剪与噪声调度由服务端下发参数


def _state_to_tensors(state: Dict[str, list]) -> Dict[str, torch.Tensor]:
    # list -> tensor，便于统一做范数计算与加噪
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in state.items()}


def _tensors_to_state(tensors: Dict[str, torch.Tensor]) -> Dict[str, list]:
    # tensor -> list，便于 JSON 序列化上传
    return {k: v.detach().cpu().tolist() for k, v in tensors.items()}


def _clip_state(
    delta_state: Dict[str, list], clip_norm: float
) -> Tuple[Dict[str, torch.Tensor], float]:
    # L2 裁剪（clip_norm）：限制单个客户端更新的敏感度
    tensors = _state_to_tensors(delta_state)
    total_norm_sq = 0.0
    for tensor in tensors.values():
        total_norm_sq += float(torch.sum(tensor ** 2).item())  # 累计 L2 范数平方
    total_norm = math.sqrt(total_norm_sq)

    if clip_norm > 0 and total_norm > clip_norm:
        scale = clip_norm / (total_norm + 1e-12)  # 计算裁剪缩放比例
        for key in tensors:
            # 裁剪应用：超阈值的更新按比例缩小
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
    # - clip_norm：裁剪阈值（由服务端下发，可为自适应）
    # - noise_multiplier：噪声系数（由服务端下发，可调度）
    # - sample_rate：batch_size / num_samples
    # - steps：本轮本地训练步数
    tensors, total_norm = _clip_state(delta_state, clip_norm)
    clipped = clip_norm > 0 and total_norm > clip_norm

    if noise_multiplier > 0:
        std = noise_multiplier * clip_norm  # 噪声标准差 = sigma * C
        for key in tensors:
            noise = torch.normal(mean=0.0, std=std, size=tensors[key].shape)  # 采样高斯噪声
            # 加噪应用：对裁剪后的更新叠加高斯噪声
            tensors[key] = tensors[key] + noise

    epsilon = 0.0
    if noise_multiplier > 0 and delta > 0:
        # 简化版 ε 估算，仅用于客户端侧统计与可视化
        epsilon = (
            sample_rate
            * math.sqrt(2.0 * steps * math.log(1.0 / delta))
            / noise_multiplier
        )

    return _tensors_to_state(tensors), float(epsilon), float(total_norm), clipped
