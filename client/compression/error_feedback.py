from typing import Dict

import torch


class ErrorFeedbackAgent:
    def __init__(self) -> None:
        self._residual_state: Dict[str, list] = {}

    def apply(self, delta_state: Dict[str, list]) -> Dict[str, list]:
        combined: Dict[str, list] = {}
        for key, value in delta_state.items():
            delta_tensor = torch.tensor(value, dtype=torch.float32)
            if key in self._residual_state:
                residual_tensor = torch.tensor(self._residual_state[key], dtype=torch.float32)
            else:
                residual_tensor = torch.zeros_like(delta_tensor)
            combined[key] = (delta_tensor + residual_tensor).cpu().tolist()
        return combined

    def update(self, combined_state: Dict[str, list], reconstructed_state: Dict[str, list]) -> None:
        new_residual: Dict[str, list] = {}
        for key, value in combined_state.items():
            combined_tensor = torch.tensor(value, dtype=torch.float32)
            if key in reconstructed_state:
                recon_tensor = torch.tensor(reconstructed_state[key], dtype=torch.float32)
            else:
                recon_tensor = torch.zeros_like(combined_tensor)
            new_residual[key] = (combined_tensor - recon_tensor).cpu().tolist()
        self._residual_state = new_residual
