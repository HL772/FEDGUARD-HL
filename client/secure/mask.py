import hashlib
from typing import Dict, List

import numpy as np
import torch

# SecureMaskingModule（AGENT.md 3.2.L）：pairwise 掩码模拟


def _pair_seed(round_id: int, client_id: str, peer_id: str, param_name: str) -> int:
    # 为每对客户端 + 参数生成稳定随机种子
    first, second = sorted([client_id, peer_id])
    raw = f"{round_id}:{first}:{second}:{param_name}".encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:4], "big")


def _mask_for_pair(
    shape: torch.Size, round_id: int, client_id: str, peer_id: str, param_name: str, scale: float
) -> torch.Tensor:
    # 生成某一对客户端的掩码向量
    seed = _pair_seed(round_id, client_id, peer_id, param_name)
    rng = np.random.default_rng(seed)
    mask = rng.normal(loc=0.0, scale=scale, size=tuple(shape))
    return torch.tensor(mask, dtype=torch.float32)


class SecureMaskingModule:
    def __init__(self, mask_scale: float = 1.0) -> None:
        self.mask_scale = mask_scale

    def apply_mask(
        self,
        delta_state: Dict[str, list],
        participants: List[str],
        client_id: str,
        round_id: int,
    ) -> Dict[str, list]:
        # 对更新做 pairwise 掩码，保证服务端只见聚合和
        masked: Dict[str, list] = {}
        for name, values in delta_state.items():
            tensor = torch.tensor(values, dtype=torch.float32)
            mask_sum = torch.zeros_like(tensor)
            for peer_id in participants:
                if peer_id == client_id:
                    continue
                mask = _mask_for_pair(
                    tensor.shape,
                    round_id,
                    client_id,
                    peer_id,
                    name,
                    self.mask_scale,
                )
                if client_id < peer_id:
                    mask_sum += mask
                else:
                    mask_sum -= mask
            masked[name] = (tensor + mask_sum).cpu().tolist()
        return masked
