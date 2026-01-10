from typing import Dict, List

import torch


class SecureAggregationAgent:
    # 安全聚合：对掩码更新求和，掩码在求和后抵消
    def aggregate(self, masked_updates: List[Dict[str, list]]) -> Dict[str, list]:
        if not masked_updates:
            raise ValueError("No masked updates to aggregate")
        first = masked_updates[0]
        agg_state: Dict[str, torch.Tensor] = {}
        for key, value in first.items():
            agg_state[key] = torch.zeros_like(torch.tensor(value, dtype=torch.float32))

        for update in masked_updates:
            for key, value in update.items():
                agg_state[key] += torch.tensor(value, dtype=torch.float32)

        return {k: v.cpu().tolist() for k, v in agg_state.items()}
