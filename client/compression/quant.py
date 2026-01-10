from typing import List, Tuple

import torch

# 简单对称量化（8-bit / 16-bit）


def quantize(values: torch.Tensor, num_bits: int) -> Tuple[float, List[int]]:
    # 将浮点数压缩成整数 + scale
    if num_bits not in (8, 16):
        raise ValueError("quant_bits must be 8 or 16")
    qmax = 2 ** (num_bits - 1) - 1
    max_abs = float(values.abs().max().item()) if values.numel() > 0 else 0.0
    if max_abs == 0.0:
        scale = 1.0
        qvalues = torch.zeros_like(values, dtype=torch.int32)
    else:
        scale = max_abs / qmax
        qvalues = torch.clamp(torch.round(values / scale), -qmax, qmax).to(torch.int32)
    return float(scale), qvalues.cpu().tolist()


def dequantize(qvalues: List[int], scale: float) -> torch.Tensor:
    # 恢复为浮点数
    return torch.tensor(qvalues, dtype=torch.float32) * float(scale)
