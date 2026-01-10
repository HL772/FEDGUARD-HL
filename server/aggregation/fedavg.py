from typing import Dict, List

import torch
from torch import nn


class FedPerModel(nn.Module):
    # 简化版 MNIST MLP（FedPer / FedPer-dual 所需的 backbone + head + private_head）
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
        )
        self.head = nn.Linear(128, 10)
        self.private_head = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

from server.aggregation.robust import (
    bulyan_aggregate,
    krum_aggregate,
    median_aggregate,
    trimmed_mean_aggregate,
)


def create_model() -> nn.Module:
    # 统一模型工厂：用于服务端初始化/评估
    return FedPerModel()


def state_dict_to_list(state_dict: Dict[str, torch.Tensor]) -> Dict[str, list]:
    # 转换为可 JSON 序列化的 list 形式
    return {k: v.detach().cpu().tolist() for k, v in state_dict.items()}


def init_model_state() -> Dict[str, list]:
    # 初始化全局模型参数（round 0）
    model = create_model()
    return state_dict_to_list(model.state_dict())


def apply_delta(base_state: Dict[str, list], delta_state: Dict[str, list]) -> Dict[str, list]:
    # 应用完整 delta（FedAvg 更新）
    updated: Dict[str, list] = {}
    for key, base_value in base_state.items():
        base_tensor = torch.tensor(base_value, dtype=torch.float32)
        delta_tensor = torch.tensor(delta_state[key], dtype=torch.float32)
        updated[key] = (base_tensor + delta_tensor).cpu().tolist()
    return updated


def apply_delta_partial(
    base_state: Dict[str, list], delta_state: Dict[str, list]
) -> Dict[str, list]:
    # 只对部分参数应用 delta（FedPer 仅聚合 backbone）
    updated: Dict[str, list] = {}
    for key, base_value in base_state.items():
        if key not in delta_state:
            updated[key] = base_value
            continue
        base_tensor = torch.tensor(base_value, dtype=torch.float32)
        delta_tensor = torch.tensor(delta_state[key], dtype=torch.float32)
        updated[key] = (base_tensor + delta_tensor).cpu().tolist()
    return updated


class AggregationModule:
    # AggregationModule（AGENT.md 3.1.C）：FedAvg + 鲁棒聚合
    def aggregate(
        self,
        updates: List[Dict[str, object]],
        use_delta: bool = False,
        method: str = "fedavg",
        trim_ratio: float = 0.2,
        byzantine_f: int = 1,
    ) -> Dict[str, list]:
        if not updates:
            raise ValueError("No updates to aggregate")
        state_key = "delta_state" if use_delta else "model_state"
        method = method.lower()
        if method == "median":
            return median_aggregate([update[state_key] for update in updates])
        if method == "trimmed_mean":
            return trimmed_mean_aggregate([update[state_key] for update in updates], trim_ratio)
        if method == "krum":
            return krum_aggregate([update[state_key] for update in updates], byzantine_f)
        if method == "bulyan":
            return bulyan_aggregate([update[state_key] for update in updates], byzantine_f)

        total_samples = sum(int(update["num_samples"]) for update in updates)
        use_equal = total_samples <= 0
        if use_equal:
            total_samples = len(updates)

        first_state = updates[0][state_key]
        agg_state: Dict[str, torch.Tensor] = {}
        for key, value in first_state.items():
            agg_state[key] = torch.zeros_like(torch.tensor(value, dtype=torch.float32))

        for update in updates:
            if use_equal:
                weight = 1.0 / total_samples
            else:
                weight = int(update["num_samples"]) / total_samples
            for key, value in update[state_key].items():
                agg_state[key] += torch.tensor(value, dtype=torch.float32) * weight

        return {k: v.cpu().tolist() for k, v in agg_state.items()}
