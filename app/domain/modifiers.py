import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from engine.interfaces import TransitionModifier


class HemophiliaMortalityModifier(TransitionModifier):
    """
    Applies age-dependent background mortality only to specified alive states.
    """

    def __init__(
        self,
        mortality_func: Callable[[int], float],
        start_age: int = 1,
        dead_state: str = "dead",
        adjust_only_states: list[str] | None = None,
        original_rate_per_1000: float = 7.76,
        enable_logger: bool = False,
    ):
        self.mortality_func = mortality_func
        self.start_age = start_age
        self.dead_state = dead_state
        self.adjust_only_states = adjust_only_states or [
            "Healthy",
            "Bleeding",
            "Hemarthrosis",
        ]
        self.enable_logger = enable_logger

        # Pre-compute original weekly mortality
        annual_p = original_rate_per_1000 / 1000
        if annual_p >= 1.0:
            self.original_weekly_mort = 1.0
        else:
            h = -np.log(1 - annual_p)
            self.original_weekly_mort = 1 - np.exp(-h / 52)

    def adjust_transition(
        self,
        base_probs: np.ndarray,  # must match
        current_state: str,  # must match
        current_chain_name: str,  # ← add this (even if unused)
        step: int,  # must match
        states: list[str],  # must match
        **kwargs: Any  # must match
    ) -> np.ndarray:
        """Runtime adjustment for age-specific background mortality."""

        if current_state not in self.adjust_only_states:
            return base_probs.copy()

        # Apply only on yearly boundaries (weekly model)
        if step % 52 != 0:
            return base_probs.copy()

        current_age = int(self.start_age + step / 52)
        annual_rate_per_1000 = self.mortality_func(current_age)
        if annual_rate_per_1000 <= 0:
            return base_probs.copy()

        annual_h = -np.log(1 - min(annual_rate_per_1000 / 1000, 0.999999))
        weekly_background = 1 - np.exp(-annual_h / 52)

        try:
            dead_idx = states.index(self.dead_state)
        except ValueError:
            return base_probs.copy()

        new_probs = base_probs.copy()
        new_probs[dead_idx] += weekly_background

        # Competing risk scaling of other probabilities
        non_death_sum = new_probs.sum() - new_probs[dead_idx]
        if non_death_sum > 0:
            scaling = (1.0 - new_probs[dead_idx]) / non_death_sum
            mask = np.arange(len(new_probs)) != dead_idx
            new_probs[mask] *= scaling

        # Final safety normalization
        total = new_probs.sum()
        if not np.isclose(total, 1.0, rtol=1e-8):
            new_probs /= total

        if self.enable_logger and not np.isclose(
            base_probs[dead_idx], new_probs[dead_idx]
        ):
            logging.info(
                "Mortality adjusted for %s at age %d (step %d, chain=%s): %.6f → %.6f",
                current_state,
                current_age,
                step,
                current_chain_name,
                base_probs[dead_idx],
                new_probs[dead_idx],
            )

        return new_probs
