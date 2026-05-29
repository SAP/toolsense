"""
EvalGenEnv — shared state schema for the realistic benchmark generation pipeline.

Per-tier state is owned by exactly one branch; reducers handle the fan-in merge
without conflicting updates.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Custom reducers
# ---------------------------------------------------------------------------

def _merge_samples(a: list, b: list) -> list:
    seen = {s["sample_id"] for s in a}
    return a + [s for s in b if s["sample_id"] not in seen]


def _keep_last(a: Any, b: Any) -> Any:
    return b


def _non_empty_list(a: list, b: list) -> list:
    return b if b else a


def _bool_or(a: bool, b: bool) -> bool:
    return a or b


def _int_max(a: int, b: int) -> int:
    return max(a, b)


def _non_none(a: Any, b: Any) -> Any:
    return b if b is not None else a


def _non_empty_str(a: str, b: str) -> str:
    return b if b else a


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class EvalGenEnv(TypedDict):
    # Identity
    id: Annotated[str, _keep_last]
    parent_id: Annotated[Optional[str], _keep_last]

    # Shared context — written once by shared_context, read-only after
    seed_tool: Annotated[dict, _keep_last]
    candidate_pool: Annotated[list, _keep_last]

    # Aggregated completed samples across all tiers
    completed_samples: Annotated[list, _merge_samples]

    # --- easy tier ---
    easy_raw_samples: Annotated[list, _non_empty_list]
    easy_generation_error: Annotated[Optional[str], _non_none]
    easy_done: Annotated[bool, _bool_or]
    easy_samples_count: Annotated[int, _int_max]
    easy_retry_count: Annotated[int, _int_max]
    easy_generation_feedback: Annotated[str, _non_empty_str]

    # --- medium tier ---
    medium_raw_samples: Annotated[list, _non_empty_list]
    medium_generation_error: Annotated[Optional[str], _non_none]
    medium_done: Annotated[bool, _bool_or]
    medium_samples_count: Annotated[int, _int_max]
    medium_retry_count: Annotated[int, _int_max]
    medium_generation_feedback: Annotated[str, _non_empty_str]

    # --- hard tier ---
    hard_raw_samples: Annotated[list, _non_empty_list]
    hard_generation_error: Annotated[Optional[str], _non_none]
    hard_done: Annotated[bool, _bool_or]
    hard_samples_count: Annotated[int, _int_max]
    hard_retry_count: Annotated[int, _int_max]
    hard_generation_feedback: Annotated[str, _non_empty_str]


def make_initial_state(seed_tool: dict) -> dict:
    """Build a fully-initialized state dict for pipeline.invoke()."""
    import uuid
    return {
        "id": str(uuid.uuid1()),
        "parent_id": None,
        "seed_tool": seed_tool,
        "candidate_pool": [],
        "completed_samples": [],
        "easy_raw_samples": [],
        "easy_generation_error": None,
        "easy_done": False,
        "easy_samples_count": 0,
        "easy_retry_count": 0,
        "easy_generation_feedback": "",
        "medium_raw_samples": [],
        "medium_generation_error": None,
        "medium_done": False,
        "medium_samples_count": 0,
        "medium_retry_count": 0,
        "medium_generation_feedback": "",
        "hard_raw_samples": [],
        "hard_generation_error": None,
        "hard_done": False,
        "hard_samples_count": 0,
        "hard_retry_count": 0,
        "hard_generation_feedback": "",
    }
