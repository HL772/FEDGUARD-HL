from typing import Any, Dict

import torch

from client.compression.quant import dequantize, quantize


def compress_state(delta_state: Dict[str, list], topk_ratio: float, quant_bits: int) -> Dict[str, Any]:
    if topk_ratio <= 0 or topk_ratio > 1:
        raise ValueError("topk_ratio must be in (0, 1]")
    payload: Dict[str, Any] = {
        "method": "topk_quant",
        "topk_ratio": float(topk_ratio),
        "quant_bits": int(quant_bits),
        "params": {},
    }
    total_elems = 0
    kept_elems = 0
    for name, values in delta_state.items():
        tensor = torch.tensor(values, dtype=torch.float32)
        flat = tensor.flatten()
        numel = int(flat.numel())
        total_elems += numel
        k = max(1, int(numel * topk_ratio))
        if k >= numel:
            indices = torch.arange(numel)
            selected = flat
        else:
            _, indices = torch.topk(flat.abs(), k)
            selected = flat[indices]
        scale, qvalues = quantize(selected, quant_bits)
        payload["params"][name] = {
            "shape": list(tensor.shape),
            "numel": numel,
            "indices": indices.cpu().tolist(),
            "values": qvalues,
            "scale": scale,
        }
        kept_elems += len(qvalues)
    payload["stats"] = {"total_elems": total_elems, "kept_elems": kept_elems}
    return payload


def decompress_state(payload: Dict[str, Any]) -> Dict[str, list]:
    if payload.get("method") != "topk_quant":
        raise ValueError("unsupported compression method")
    params = payload.get("params", {})
    state: Dict[str, list] = {}
    for name, item in params.items():
        numel = int(item["numel"])
        shape = list(item["shape"])
        indices = torch.tensor(item["indices"], dtype=torch.long)
        values = dequantize(item["values"], float(item["scale"]))
        flat = torch.zeros(numel, dtype=torch.float32)
        if indices.numel() > 0:
            flat[indices] = values
        tensor = flat.view(*shape)
        state[name] = tensor.cpu().tolist()
    return state


class CompressionAgent:
    def __init__(self, topk_ratio: float, quant_bits: int) -> None:
        self.topk_ratio = topk_ratio
        self.quant_bits = quant_bits

    def compress(self, delta_state: Dict[str, list]) -> Dict[str, Any]:
        return compress_state(delta_state, self.topk_ratio, self.quant_bits)

    def decompress(self, payload: Dict[str, Any]) -> Dict[str, list]:
        return decompress_state(payload)
