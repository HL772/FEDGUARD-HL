import math
from typing import List


# RDP 会计：累计 DP ε 预算
# - 目标：把“每轮噪声 + 采样率”的隐私损耗累积成全局 ε
# - 采用 RDP（Rényi Differential Privacy）近似，再转换为 (ε, δ)-DP
class RDPAccountant:
    def __init__(self, orders: List[float] | None = None) -> None:
        # orders：RDP 的阶数 α；多阶数便于取最优 ε
        if orders is None:
            orders = [1.5, 2, 3, 5, 10, 20, 40, 80]
        self._orders = orders
        # _rdp：每个 α 的累计 RDP 值
        self._rdp = [0.0 for _ in orders]

    @property
    def orders(self) -> List[float]:
        # 对外暴露阶数列表（拷贝，避免外部修改）
        return list(self._orders)

    def update(self, sample_rate: float, noise_multiplier: float, steps: int = 1) -> None:
        # 更新 RDP 预算（每轮调用一次）
        # - sample_rate：采样率 q（≈ batch_size / num_samples）
        # - noise_multiplier：噪声系数 σ
        # - steps：本轮本地训练步数（影响隐私累计量）
        if sample_rate <= 0 or noise_multiplier <= 0 or steps <= 0:
            return
        for idx, order in enumerate(self._orders):
            # 每个 α 的 RDP 累加：RDP_total += steps * RDP_step
            self._rdp[idx] += steps * _approx_rdp(order, sample_rate, noise_multiplier)

    def get_epsilon(self, delta: float) -> float:
        # 将累计 RDP 转换为 (ε, δ)-DP
        # 公式：ε = RDP - log(δ) / (α - 1)
        # 对所有 α 取最小 ε
        if delta <= 0:
            return 0.0
        eps_values = []
        for rdp, order in zip(self._rdp, self._orders):
            if order <= 1:
                continue
            eps = rdp - math.log(delta) / (order - 1.0)
            eps_values.append(eps)
        return float(min(eps_values)) if eps_values else 0.0


def _approx_rdp(order: float, sample_rate: float, noise_multiplier: float) -> float:
    # 子采样高斯机制的 RDP 近似
    # - order：RDP 阶数 α
    # - sample_rate：采样率 q
    # - noise_multiplier：噪声系数 σ
    sigma_sq = noise_multiplier ** 2
    if sample_rate >= 1.0:
        # 全采样近似：α / (2σ^2)
        return float(order) / (2.0 * sigma_sq)
    # 子采样近似：q^2 * α / (2σ^2)
    return (sample_rate ** 2) * float(order) / (2.0 * sigma_sq)
