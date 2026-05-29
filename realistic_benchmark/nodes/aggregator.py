"""Aggregator node — sorts completed samples by complexity tier after fan-in."""


class AggregatorNode:
    _COMPLEXITY_ORDER = {"easy": 0, "medium": 1, "hard": 2}

    def __call__(self, state: dict) -> dict:
        samples = list(state.get("completed_samples", []))
        samples.sort(key=lambda s: self._COMPLEXITY_ORDER.get(s["complexity"], 99))
        return {"completed_samples": samples}
