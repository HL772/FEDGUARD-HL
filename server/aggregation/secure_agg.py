from typing import Dict, List

import torch


class SecureAggregationModule:
    # 安全聚合：对掩码更新求和，掩码在求和后抵消
    def aggregate(self, masked_updates: List[Dict[str, list]]) -> Dict[str, list]:
        # 输入：各客户端上传的 masked_update（已包含 pairwise 掩码）
        if not masked_updates:
            raise ValueError("No masked updates to aggregate")
        first = masked_updates[0]
        agg_state: Dict[str, torch.Tensor] = {}
        for key, value in first.items():
            # 为每个参数创建累加缓冲区
            agg_state[key] = torch.zeros_like(torch.tensor(value, dtype=torch.float32))

        for update in masked_updates:
            for key, value in update.items():
                # 逐客户端累加掩码更新（mask 会在总和中互相抵消）
                agg_state[key] += torch.tensor(value, dtype=torch.float32)

        # 输出：聚合和（不含任何单客户端明文更新）
        return {k: v.cpu().tolist() for k, v in agg_state.items()}
