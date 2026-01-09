from server.aggregation.robust import bulyan_aggregate, krum_aggregate


def _make_state(value: float) -> dict:
    return {"w": [value]}


def test_krum_aggregate_filters_outlier() -> None:
    states = [_make_state(0.1) for _ in range(4)]
    states.append(_make_state(10.0))
    aggregated = krum_aggregate(states, byzantine_f=1)
    assert aggregated["w"][0] < 1.0


def test_bulyan_aggregate_filters_outlier() -> None:
    states = [_make_state(0.2) for _ in range(7)]
    states.append(_make_state(12.0))
    aggregated = bulyan_aggregate(states, byzantine_f=1)
    assert aggregated["w"][0] < 1.0
