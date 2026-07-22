from typing import Any, Protocol

import numpy as np


class TransitionModifier(Protocol):
    """Lightweight protocol for runtime modifiers (mortality, switching logic, etc.)."""

    def adjust_transition(
        self,
        base_probs: np.ndarray,
        current_state: str,
        current_chain_name: str,
        step: int,
        states: list[str],
        **kwargs: Any,
    ) -> np.ndarray:
        """Return adjusted probabilities (must still sum to ~1.0)."""
        ...


class NoOpModifier:
    """Default do-nothing modifier."""

    def adjust_transition(
        self,
        base_probs: np.ndarray,
        current_state: str,
        current_chain_name: str,
        step: int,
        states: list[str],
        **kwargs: Any,
    ) -> np.ndarray:
        return base_probs.copy()
