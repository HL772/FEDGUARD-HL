import math
from typing import List


class RDPAccountant:
    def __init__(self, orders: List[float] | None = None) -> None:
        if orders is None:
            orders = [1.5, 2, 3, 5, 10, 20, 40, 80]
        self._orders = orders
        self._rdp = [0.0 for _ in orders]

    @property
    def orders(self) -> List[float]:
        return list(self._orders)

    def update(self, sample_rate: float, noise_multiplier: float, steps: int = 1) -> None:
        if sample_rate <= 0 or noise_multiplier <= 0 or steps <= 0:
            return
        for idx, order in enumerate(self._orders):
            self._rdp[idx] += steps * _approx_rdp(order, sample_rate, noise_multiplier)

    def get_epsilon(self, delta: float) -> float:
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
    sigma_sq = noise_multiplier ** 2
    if sample_rate >= 1.0:
        return float(order) / (2.0 * sigma_sq)
    return (sample_rate ** 2) * float(order) / (2.0 * sigma_sq)
