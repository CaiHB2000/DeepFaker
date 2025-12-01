from __future__ import annotations


def linear_warmup(
    current_step: int,
    warmup_steps: int,
    max_value: float,
) -> float:
    if warmup_steps <= 0:
        return max_value
    progress = min(max(current_step / warmup_steps, 0.0), 1.0)
    return progress * max_value
