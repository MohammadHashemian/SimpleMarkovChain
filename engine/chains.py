from collections import OrderedDict
from collections.abc import Callable, Generator
from dataclasses import asdict
from typing import (
    Any,
    Protocol,
    TypeVar,
)

import numpy as np

from engine.modifier import NoOpModifier, TransitionModifier

_NOOP_MODIFIER = NoOpModifier()

T = TypeVar("T")
U = TypeVar("U")


class Chain:
    def __init__(self, name: str, states: list[str], matrix: np.ndarray) -> None:
        self.name = name
        self.states = states
        self.matrix = matrix

        if matrix.shape != (len(states), len(states)):
            raise ValueError("Transition matrix must be square (n_states x n_states)")

    def update(self, matrix: np.ndarray):
        self.matrix = matrix


class MarkovModel(Protocol):
    def run(self) -> list[str]: ...
    def collect_rewards(self) -> dict[str, list[float | int]]: ...
    def add_reward_function(self, func: Callable) -> None: ...
    def add_store_function(self, arg_name: str, func: Callable) -> None: ...


class MarkovChains:
    """
    Summary
    --------
    Generic Markov chain with support for:
      - Multiple named chains + dynamic switching via conditions
      - Reward / store functions
      - Pluggable transition modifiers (for domain-specific adjustments like mortality)
      - Absorbing state detection (e.g. death) with early stopping

    Args:
        chains: List of chain objects
        entrance: Initial state for the chain to start with
        entrance_chain: Name of the initial chain to use
        steps: Number of steps to simulate
        conditions: Dictionary containing target chain name and the switch condition function
        worker_kwargs: Keyword arguments which will be passed to reward and switch functions on call
        transition_modifier: Optional modifier applied to transition probabilities each step
        absorbing_states: Set of absorbing states. If None, auto-detects
            states with probability 1 to stay.
    """

    def __init__(
        self,
        chains: list["Chain"],
        entrance: str,
        entrance_chain: str,
        steps: int,
        conditions: dict[str, Callable] | None = None,
        worker_kwargs: dict[str, Any] | None = None,
        transition_modifier: TransitionModifier | None = None,
        absorbing_states: set[str] | None = None,
    ) -> None:
        self.steps = steps
        self.chains = chains
        self._chains_map = {chain.name: chain for chain in chains}
        self.conditions = conditions or {}
        self._has_conditions = bool(self.conditions)
        self.worker_kwargs = worker_kwargs or {}
        self.transition_modifier = transition_modifier or _NOOP_MODIFIER
        self._is_noop_modifier = type(self.transition_modifier) is NoOpModifier

        self.absorbing_states = set(absorbing_states) if absorbing_states else set()
        self._auto_detect_absorbing = absorbing_states is None

        self._reward_functions: list[Callable] = []
        self._rewards: dict[str, list[float | int]] = {}
        self._store: dict[str, Callable] = OrderedDict()

        self.entrance = entrance
        self._current_chain_name = entrance_chain
        self.current_state_idx = self._chains_map[entrance_chain].states.index(entrance)

        # Metadata
        self.absorbed_at: int | None = None
        self.absorbed_state: str | None = None

        # Validation
        if entrance_chain not in self._chains_map:
            raise ValueError(f"Start chain '{entrance_chain}' not found")
        if entrance not in self._chains_map[entrance_chain].states:
            raise ValueError(
                f"Start state '{entrance}' not in chain '{entrance_chain}'"
            )
        for chain in self.chains:
            if chain.matrix.shape != (len(chain.states), len(chain.states)):
                raise ValueError(f"Chain '{chain.name}' matrix shape mismatch")
            if not np.allclose(chain.matrix.sum(axis=1), 1.0, rtol=1e-5):
                raise ValueError(f"Chain '{chain.name}' rows must sum to 1")

        # Validate user-provided absorbing states
        if self.absorbing_states:
            for state in self.absorbing_states:
                if not any(state in chain.states for chain in self.chains):
                    raise ValueError(
                        f"Absorbing state '{state}' not found in any chain"
                    )

        # Pre-compute boolean absorbing mask per chain: states[i] is absorbing -> mask[i] = True
        # Eliminates per-step chain.states.index(state) + np.isclose checks in _is_absorbing.
        self._absorbing_mask: dict[str, np.ndarray] = {}
        for chain in self.chains:
            n = len(chain.states)
            mask = np.zeros(n, dtype=bool)
            matrix = chain.matrix
            for i in range(n):
                row = matrix[i]
                is_user_absorbing = chain.states[i] in self.absorbing_states
                is_auto_absorbing = (
                    self._auto_detect_absorbing
                    and len(row) > 0
                    and np.isclose(row[i], 1.0, rtol=1e-8)
                    and np.allclose(
                        row,
                        np.array([1.0 if j == i else 0.0 for j in range(n)]),
                        rtol=1e-8,
                    )
                )
                if is_user_absorbing or is_auto_absorbing:
                    mask[i] = True
            self._absorbing_mask[chain.name] = mask

    def _is_absorbing(self, state: str, chain_name: str) -> bool:
        """Check if a state is absorbing in the current chain (mask lookup)."""
        chain = self._chains_map[chain_name]
        try:
            idx = chain.states.index(state)
        except ValueError:
            return False
        return bool(self._absorbing_mask[chain_name][idx])

    def _worker_kwargs_dict(self) -> dict:
        """Return worker kwargs as a dict whether originally a dict or a dataclass/object."""
        if isinstance(self.worker_kwargs, dict):
            return self.worker_kwargs
        try:
            return asdict(self.worker_kwargs)
        except Exception:
            try:
                return dict(vars(self.worker_kwargs))
            except Exception:
                return {}

    def _get_rng(self) -> np.random.Generator:
        """Return a per-call numpy Generator.

        Looks up ``worker_kwargs['rng']`` (the contract used by ``worker_function``
        to pass a seeded ``np.random.default_rng`` per worker) and falls back to a
        fresh local generator when none is provided. This guarantees that the
        state transition draws are always driven by a Generator that the caller
        can seed, never by the global ``np.random`` state.
        """
        kwargs = self.worker_kwargs
        rng = None
        if isinstance(kwargs, dict):
            rng = kwargs.get("rng")
        else:
            for attr in ("rng", "_rng", "random_state"):
                if hasattr(kwargs, attr):
                    rng = getattr(kwargs, attr)
                    break
        if isinstance(rng, np.random.Generator):
            return rng
        return np.random.default_rng()

    def _compute_rewards(
        self, step: int, current_state: str, base_kwargs: dict[str, Any]
    ) -> None:
        """Compute store and reward functions for current step."""
        reward_kwargs: dict[str, Any] = {}
        # Store functions first
        for arg_name, func in self._store.items():
            res = func(
                state=current_state,
                chain=self._current_chain_name,
                step=step,
                **base_kwargs,
                **reward_kwargs,
            )
            reward_kwargs[arg_name] = res
            self._rewards[arg_name].append(res)

        # Normal reward functions
        for func in self._reward_functions:
            reward = func(
                step=step,
                state=current_state,
                chain=self._current_chain_name,
                **reward_kwargs,
                **base_kwargs,
            )
            self._rewards[func.__name__].append(reward)

    def add_reward_function(self, func: Callable) -> None:
        """Add a reward function to be calculated at each step."""
        self._reward_functions.append(func)
        self._rewards[func.__name__] = []

    def add_store_function(self, arg_name: str, func: Callable) -> None:
        """
        Adds a function that will be called prior to normal reward functions,
        and its results will be passed to normal reward functions.
        """
        self._store[arg_name] = func
        self._rewards[arg_name] = []

    def _get_current_chain(self) -> Chain:
        """Returns the active chain for states and transition extractions"""
        return self._chains_map[self._current_chain_name]

    def walk(self, steps: int | None = None) -> Generator[str, None, None]:
        steps = steps if steps is not None else self.steps
        current_state_idx = self.current_state_idx
        current_chain = self._get_current_chain()
        states = current_chain.states
        transitions = current_chain.matrix

        # Cache worker kwargs once per walk: avoids re-evaluating the dataclass
        # -> dict conversion on every store/reward/condition/modifier call.
        base_kwargs = self._worker_kwargs_dict()

        # Resolve a per-walk numpy Generator. The walk MUST draw transitions
        # from this Generator, never from the global np.random state, so the
        # simulator stays reproducible when a worker seeds its own rng via
        # ``worker_kwargs['rng']``. If no rng was supplied, fall back to a
        # fresh local generator (still isolated from the global state).
        rng = self._get_rng()
        _rng_choice = rng.choice

        # Bind hot attributes to locals for faster attribute access in the loop.
        absorbing_mask = self._absorbing_mask
        _chains_map = self._chains_map
        _compute_rewards = self._compute_rewards
        _is_noop_modifier = self._is_noop_modifier
        _has_conditions = self._has_conditions
        conditions_items = self.conditions.items
        current_chain_name = self._current_chain_name

        for step in range(steps + 1):
            current_state = states[current_state_idx]
            yield current_state

            # Check for absorption (mask lookup, O(1))
            if bool(absorbing_mask[current_chain_name][current_state_idx]):
                _compute_rewards(step, current_state, base_kwargs)
                self.absorbed_at = step
                self.absorbed_state = current_state
                self.current_state_idx = current_state_idx
                return  # Early exit on absorption

            # Normal step: compute rewards
            _compute_rewards(step, current_state, base_kwargs)

            if step >= steps:
                break

            # Chain switching (skipped entirely when no conditions defined)
            switched = False
            if _has_conditions:
                for chain_name, condition in conditions_items():
                    if chain_name != current_chain_name and condition(
                        step,
                        current_state,
                        current_chain_name,
                        **base_kwargs,
                    ):
                        current_chain_name = chain_name
                        self._current_chain_name = chain_name
                        current_chain = _chains_map[chain_name]
                        states = current_chain.states
                        transitions = current_chain.matrix
                        try:
                            current_state_idx = states.index(current_state)
                        except ValueError:
                            current_state_idx = 0
                        switched = True
                        break

            if switched:
                continue

            # Transition with modifier (skipped entirely for NoOpModifier)
            if _is_noop_modifier:
                base_probs = transitions[current_state_idx]
                current_state_idx = int(
                    _rng_choice(len(states), p=base_probs)
                )
            else:
                base_probs = transitions[current_state_idx]
                adjusted_probs = self.transition_modifier.adjust_transition(
                    base_probs=base_probs,
                    current_state=current_state,
                    current_chain_name=current_chain_name,
                    step=step,
                    states=states,
                    **base_kwargs,
                )
                current_state_idx = int(
                    _rng_choice(len(states), p=adjusted_probs)
                )

        self.current_state_idx = current_state_idx

    def run(self) -> list[str]:
        """Run the Markov chain and return the complete sequence of states."""
        return list(self.walk())

    def collect_rewards(self) -> dict[str, list[float | int]]:
        """Return all collected rewards for each reward function."""
        return self._rewards

    def is_absorbed(self) -> bool:
        """Return True if the simulation reached an absorbing state."""
        return self.absorbed_at is not None
