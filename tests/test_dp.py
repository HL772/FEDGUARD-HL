import unittest

from server.orchestrator import CoordinatorAgent
from server.privacy.accountant import RDPAccountant


class TestRDPAccountant(unittest.TestCase):
    def test_epsilon_increases_with_steps(self) -> None:
        accountant = RDPAccountant()
        accountant.update(sample_rate=0.1, noise_multiplier=1.0, steps=1)
        eps_1 = accountant.get_epsilon(delta=1e-5)
        accountant.update(sample_rate=0.1, noise_multiplier=1.0, steps=1)
        eps_2 = accountant.get_epsilon(delta=1e-5)
        self.assertGreater(eps_2, eps_1)

    def test_epsilon_decreases_with_more_noise(self) -> None:
        accountant_low = RDPAccountant()
        accountant_low.update(sample_rate=0.1, noise_multiplier=1.0, steps=1)
        eps_low = accountant_low.get_epsilon(delta=1e-5)

        accountant_high = RDPAccountant()
        accountant_high.update(sample_rate=0.1, noise_multiplier=2.0, steps=1)
        eps_high = accountant_high.get_epsilon(delta=1e-5)
        self.assertLess(eps_high, eps_low)


class TestAdaptiveClip(unittest.TestCase):
    def test_clip_norm_ema_updates(self) -> None:
        dp_config = {
            "enabled": True,
            "mode": "adaptive_rdp",
            "clip_norm": 2.0,
            "adaptive_clip": {"enabled": True, "percentile": 0.5, "ema": 0.5},
        }
        coordinator = CoordinatorAgent(
            max_rounds=1,
            clients_per_round=1,
            dp_config=dp_config,
        )
        first = coordinator._update_clip_norm([1.0, 3.0, 5.0])
        self.assertAlmostEqual(first, 2.5, places=5)
        second = coordinator._update_clip_norm([2.0, 2.0, 2.0])
        self.assertAlmostEqual(second, 2.25, places=5)


if __name__ == "__main__":
    unittest.main()
