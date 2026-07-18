from typing import (
    List,
    Generator,
    Optional,
    Callable,
    Dict,
    Any,
    Set,
    TypeVar,
    Protocol,
)
from dataclasses import asdict
from collections import OrderedDict
from engine.interfaces import TransitionModifier
from engine.interfaces import NoOpModifier
import numpy as np

T = TypeVar("T")
U = TypeVar("U")


class Chain:
    def __init__(self, name: str, states: List[str], matrix: np.ndarray) -> None:
        self.name = name
        self.states = states
        self.matrix = matrix

        if matrix.shape != (len(states), len(states)):
            raise ValueError("Transition matrix must be square (n_states x n_states)")

    def update(self, matrix: np.ndarray):
        self.matrix = matrix


class MarkovModel(Protocol):
    def run(self) -> List[str]: ...
    def collect_rewards(self) -> Dict[str, List[float | int]]: ...
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
        absorbing_states: Set of absorbing states. If None, auto-detects states with probability 1 to stay.
    """

    def __init__(
        self,
        chains: List["Chain"],
        entrance: str,
        entrance_chain: str,
        steps: int,
        conditions: Optional[Dict[str, Callable]] = None,
        worker_kwargs: Optional[Dict[str, Any]] = None,
        transition_modifier: Optional[TransitionModifier] = None,
        absorbing_states: Optional[Set[str]] = None,
    ) -> None:
        self.steps = steps
        self.chains = chains
        self._chains_map = {chain.name: chain for chain in chains}
        self.conditions = conditions or {}
        self.worker_kwargs = worker_kwargs or {}
        self.transition_modifier = transition_modifier or NoOpModifier()

        self.absorbing_states = set(absorbing_states) if absorbing_states else set()
        self._auto_detect_absorbing = absorbing_states is None

        self._reward_functions: List[Callable] = []
        self._rewards: Dict[str, List[float | int]] = {}
        self._store: Dict[str, Callable] = OrderedDict()

        self.entrance = entrance
        self._current_chain_name = entrance_chain
        self.current_state_idx = self._chains_map[entrance_chain].states.index(entrance)

        # Metadata
        self.absorbed_at: Optional[int] = None
        self.absorbed_state: Optional[str] = None

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

    def _is_absorbing(self, state: str, chain_name: str) -> bool:
        """Check if a state is absorbing in the current chain."""
        if state in self.absorbing_states:
            return True

        if not self._auto_detect_absorbing:
            return False

        chain = self._chains_map[chain_name]
        try:
            idx = chain.states.index(state)
            row = chain.matrix[idx]
            # True absorbing state: probability 1.0 to stay in itself
            return (
                len(row) > 0
                and np.isclose(row[idx], 1.0, rtol=1e-8)
                and np.allclose(
                    row, [1.0 if i == idx else 0.0 for i in range(len(row))], rtol=1e-8
                )
            )
        except (ValueError, IndexError):
            return False

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

    def _compute_rewards(self, step: int, current_state: str) -> None:
        """Compute store and reward functions for current step."""
        reward_kwargs = {}
        # Store functions first
        for arg_name, func in self._store.items():
            res = func(
                state=current_state,
                chain=self._current_chain_name,
                step=step,
                **self._worker_kwargs_dict(),
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
                **self._worker_kwargs_dict(),
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

    def walk(self, steps: Optional[int] = None) -> Generator[str, None, None]:
        steps = steps if steps is not None else self.steps
        current_state_idx = self.current_state_idx
        current_chain = self._get_current_chain()
        states = current_chain.states
        transitions = current_chain.matrix

        for step in range(steps + 1):
            current_state = states[current_state_idx]
            yield current_state

            # Check for absorption
            if self._is_absorbing(current_state, self._current_chain_name):
                self._compute_rewards(step, current_state)
                self.absorbed_at = step
                self.absorbed_state = current_state
                self.current_state_idx = current_state_idx
                return  # Early exit on absorption

            # Normal step: compute rewards
            self._compute_rewards(step, current_state)

            if step >= steps:
                break

            # Chain switching
            switched = False
            for chain_name, condition in self.conditions.items():
                if chain_name != self._current_chain_name and condition(
                    step,
                    current_state,
                    self._current_chain_name,
                    **self._worker_kwargs_dict(),
                ):
                    self._current_chain_name = chain_name
                    current_chain = self._get_current_chain()
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

            # Transition with modifier
            base_probs = transitions[current_state_idx]
            adjusted_probs = self.transition_modifier.adjust_transition(
                base_probs=base_probs,
                current_state=current_state,
                current_chain_name=self._current_chain_name,
                step=step,
                states=states,
                **self._worker_kwargs_dict(),
            )

            # Safety normalization
            # adjusted_probs = np.array(adjusted_probs, dtype=float)
            # total = adjusted_probs.sum()
            # if not np.isclose(total, 1.0, rtol=1e-8):
            #     adjusted_probs /= total

            current_state_idx = np.random.choice(len(states), p=adjusted_probs)

        self.current_state_idx = current_state_idx

    def run(self) -> List[str]:
        """Run the Markov chain and return the complete sequence of states."""
        return list(self.walk())

    def collect_rewards(self) -> Dict[str, List[float | int]]:
        """Return all collected rewards for each reward function."""
        return self._rewards

    def is_absorbed(self) -> bool:
        """Return True if the simulation reached an absorbing state."""
        return self.absorbed_at is not None
