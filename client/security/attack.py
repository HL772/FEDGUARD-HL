import math
from typing import Dict, Iterable, Optional

import torch



class AttackSimulator:
    def __init__(self, method: str = "none", scale: float = 1.0) -> None:
        self.method = str(method or "none").lower()
        self.scale = float(scale)

    def is_malicious(
        self,
        client_rank: int,
        num_clients: int,
        malicious_ranks: Optional[Iterable[int]] = None,
        malicious_fraction: float = 0.0,
    ) -> bool:
        # 判断当前客户端是否作为恶意样本参与
        if malicious_ranks:
            return client_rank in set(int(rank) for rank in malicious_ranks)
        if malicious_fraction > 0:
            cutoff = max(1, int(math.ceil(num_clients * float(malicious_fraction))))
            return client_rank < cutoff
        return False

    def apply(self, delta_state: Dict[str, list]) -> Dict[str, list]:
        # 对更新施加攻击扰动
        if self.method == "sign_flip":
            factor = -abs(self.scale or 1.0)
        elif self.method == "scale":
            factor = float(self.scale or 1.0)
        else:
            return delta_state
        attacked: Dict[str, list] = {}
        for key, value in delta_state.items():
            tensor = torch.tensor(value, dtype=torch.float32)
            attacked[key] = (tensor * factor).cpu().tolist()
        return attacked
