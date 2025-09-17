from typing import List, Union, Generator, Optional, Callable, Dict, Literal, Tuple, Any
from collections import OrderedDict
from pydantic import BaseModel
from model.utils import prob_at_least_one
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Chain:
    def __init__(self, name: str, states: List[str], matrix: np.ndarray) -> None:
        self.name = name
        self.states = states
        self.matrix = matrix

        if matrix.shape != (len(states), len(states)):
            raise ValueError("Transition matrix should be in cubic shape")


class MarkovChains:
    """
    Summary
    --------
    A Markov chain implementation that generates state transitions with reward tracking and dynamic chain switching.

    Initialize Markov chain with multiple chains, transitions, and optional reward function.

    Args:
        chains: List of chain objects
        entrance: Initial state for the chain to start with
        entrance_chain: Name of the initial chain to use
        steps: Number of steps to simulate
        conditions: Dictionary containing target chain name and the switch condition function
        worker_kwargs: Keyword arguments which will be pass to reward and switch functions on call
    """

    def __init__(
        self,
        chains: List["Chain"],
        entrance: str,
        entrance_chain: str,
        steps: int,
        conditions: Optional[Dict[str, Callable]] = None,
        worker_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.steps = steps
        self.chains = chains
        self.chains_map = {chain.name: chain for chain in chains}
        self.conditions = conditions or {}
        self.worker_kwargs = worker_kwargs or {}
        self.reward_functions: List[Callable] = []
        self.rewards: Dict[str, List[float | int]] = {}
        self.entrance = entrance
        self.current_chain_name = entrance_chain
        self.store: Dict[str, Callable] = OrderedDict()

        # Validate chains
        if entrance_chain not in self.chains_map.keys():
            raise ValueError(f"Start chain '{entrance_chain}' not in provided chains")
        if entrance not in self.chains_map[entrance_chain].states:
            raise ValueError(
                f"Start state '{entrance}' not in states list of chain '{entrance_chain}'"
            )
        for chain in self.chains:
            if chain.matrix.shape != (len(chain.states), len(chain.states)):
                raise ValueError(
                    f"Chain '{chain.name}': Expected {len(chain.states)}x{len(chain.states)} transition matrix, got {chain.matrix.shape}"
                )
            if not np.allclose(chain.matrix.sum(axis=1), 1, rtol=1e-5):
                raise ValueError(
                    f"Chain '{chain.name}': Each row in the transition matrix must sum to 1"
                )

        # Set initial state index
        self.current_state_idx = self.chains_map[entrance_chain].states.index(entrance)

    def add_reward_function(self, func: Callable) -> None:
        """Add a reward function to be calculated at each step."""
        self.reward_functions.append(func)
        self.rewards[func.__name__] = []

    def add_store_function(self, arg_name: str, func: Callable) -> None:
        """
        Adds a reward function that will be called prior to normal reward functions,
        and it's results will be passed to normal reward functions
        """
        self.store[arg_name] = func
        self.rewards[arg_name] = []

    def _get_current_chain(self) -> Chain:
        """Returns the active chain for states and transition extractions"""
        return self.chains_map[self.current_chain_name]

    def walk(self, steps: Optional[int] = None) -> Generator[str, None, None]:
        """Generate a sequence of states for the specified number of steps."""
        steps = steps if steps is not None else self.steps
        current_state_idx: int = self.current_state_idx
        current_chain: Chain = self._get_current_chain()
        states, transitions = current_chain.states, current_chain.matrix

        for step in range(self.steps + 1):  # Include final state
            yield states[current_state_idx]

            # Process store functions
            reward_kwargs = {}
            for arg, func in self.store.items():
                res = func(
                    state=states[current_state_idx],
                    chain=self.current_chain_name,
                    **self.worker_kwargs,
                    **reward_kwargs,
                )
                reward_kwargs[arg] = res
                self.rewards[arg].append(res)

            # Process reward functions
            for func in self.reward_functions:
                reward = func(
                    step=step,
                    state=states[current_state_idx],
                    chain=self.current_chain_name,
                    **reward_kwargs,
                    **self.worker_kwargs,
                )
                self.rewards[func.__name__].append(reward)

            # Skip transition for final step
            if step < self.steps:
                # Check for chain switch
                for chain_name, condition in self.conditions.items():
                    if chain_name != self.current_chain_name and condition(
                        step,
                        states[current_state_idx],
                        self.current_chain_name,
                        **self.worker_kwargs,
                    ):
                        self.current_chain_name = chain_name
                        current_active_chain = self._get_current_chain()
                        states, transitions = (
                            current_active_chain.states,
                            current_active_chain.matrix,
                        )
                        try:
                            current_state_idx = states.index(states[current_state_idx])
                        except ValueError:
                            print("State doesn't exist in new chain, reset to start")
                            current_state_idx = 0
                        break

                # Transition
                probs = transitions[current_state_idx]
                current_state_idx = np.random.choice(len(states), p=probs)

    def collect_rewards(self) -> Dict[str, List[float | int]]:
        """Return all collected rewards for each reward function."""
        return self.rewards

    def run(self) -> List[str]:
        """Run the Markov chain and return the complete sequence of states."""
        return list(self.walk())


class MarkovResult(BaseModel):
    """
    Common returned results from markov models simulations
    """

    initial_state: str
    final_state: str
    steps: int
    path: List[str]


class TransitionGenerator:
    def __init__(
        self,
        states: List[str],
        transition_pairs: Dict[
            Tuple[str, str],
            Tuple[Union[float, str], Optional[Literal["weekly", "annual"]]],
        ],
        special_transitions: Optional[Dict[str, List[float]]] = None,
        time_step: Literal["weekly", "annual"] = "weekly",
    ) -> None:
        """
        Initialize TransitionGenerator with states and transition pairs.

        Args:
            states: List of states for the transition matrix.
            transition_pairs: Dictionary of (from_state, to_state) -> (value, period) pairs.
                              Value is a probability or rate; period is 'annual', 'weekly', or None (for direct probabilities).
            special_transitions: Optional dictionary of state -> transition probabilities for states with fixed transitions.
            time_step: The time step for the transition matrix ('weekly' or 'annual').
        """
        self.states = states
        self.transition_pairs = transition_pairs
        self.special_transitions = special_transitions or {}
        self.state_indices = {state: idx for idx, state in enumerate(states)}
        self.time_step = time_step

        # Validate states
        if not states:
            raise ValueError("States list cannot be empty")
        if "Death" not in states:
            raise ValueError("Death state is required for an absorbing state")

        # Validate transition pairs
        for (from_state, to_state), (value, period) in transition_pairs.items():
            if from_state not in states or to_state not in states:
                raise ValueError(f"Invalid state in pair ({from_state}, {to_state})")
            if not isinstance(value, (int, float, str)):
                raise ValueError(f"Transition value {value} must be a number or string")
            if period not in {None, "annual", "weekly"}:
                raise ValueError(
                    f"Invalid period {period}; must be None, 'annual', or 'weekly'"
                )

        # Validate special transitions
        for state, probs in self.special_transitions.items():
            if state not in states:
                raise ValueError(f"Special transition state {state} not in states")
            if len(probs) != len(states):
                raise ValueError(
                    f"Special transition for {state} must have {len(states)} probabilities"
                )
            if not np.allclose(sum(probs), 1, rtol=1e-5):
                raise ValueError(
                    f"Special transition probabilities for {state} must sum to 1"
                )

    def get_probability(
        self,
        value: Union[float, int, str],
        period: Optional[Literal["weekly", "annual"]],
    ) -> float:
        """
        Convert a rate (event frequency) on given interval to a probability value.

        Note:
            if period value is None, use passed value directly as probability

        Args:
            value: The probability or rate.
            period: The period for rate conversion ('annual', 'weekly', or None).

        Returns:
            The probability of at least on event.
        """
        if period is None:
            return float(value)
        rate = float(value)
        if period == self.time_step:
            lam_ts = rate
        elif period == "annual" and self.time_step == "weekly":
            lam_ts = rate / 52
        elif period == "weekly" and self.time_step == "annual":
            lam_ts = rate * 52
        else:
            raise ValueError(
                f"Cannot convert from period {period} to time_step {self.time_step}"
            )
        return prob_at_least_one(lam_ts)

    def build(self) -> List[List[float]]:
        """
        Construct discrete-time approximation of a continuous-time events by
        converting continuous transition rates (λ) into discrete transition
        probabilities for a given time step.

        Handles:
            - Direct probabilities (period=None)
            - Competing risks from event rates (period="weekly" or "annual")
            - Special transitions (explicit rows)
            - Mixing direct probabilities and rates for the same state

        Returns:
            Transition matrix as a list of lists
        """
        n = len(self.states)
        matrix = [[0.0] * n for _ in range(n)]

        # --- Handle special transitions directly ---
        for state, probs in self.special_transitions.items():
            idx = self.state_indices[state]
            matrix[idx] = probs[:]  # Copy to avoid mutation

        # --- Handle regular states ---
        for state in self.states:
            if state in self.special_transitions:
                continue  # Already assigned

            idx = self.state_indices[state]
            probs = [0.0] * n

            # Collect outgoing transitions
            transitions = {
                to_state: (value, period)
                for (from_state, to_state), (
                    value,
                    period,
                ) in self.transition_pairs.items()
                if from_state == state
            }

            if not transitions:
                # No outgoing transitions → absorbing/self-loop
                probs[idx] = 1.0
                matrix[idx] = probs
                continue

            # Split transitions into direct probabilities and rates
            direct_probs = {}
            rate_transitions = {}
            for to_state, (value, period) in transitions.items():
                if period is None:
                    direct_probs[to_state] = float(value)
                else:
                    rate_transitions[to_state] = (value, period)

            # --- Handle direct probabilities ---
            total_direct_prob = sum(direct_probs.values())
            if total_direct_prob > 1.0:
                raise ValueError(
                    f"Sum of direct probabilities for {state} exceeds 1: {total_direct_prob}"
                )

            for to_state, p in direct_probs.items():
                probs[self.state_indices[to_state]] = p

            # --- Handle rate-based transitions ---
            remaining_prob = 1.0 - total_direct_prob
            if remaining_prob < 0:
                raise ValueError(
                    f"Negative remaining probability for {state}: {remaining_prob}"
                )

            if rate_transitions and remaining_prob > 0:
                lam_dict = {}
                for to_state, (value, period) in rate_transitions.items():
                    rate = float(value)
                    if period == self.time_step:
                        lam = rate
                    elif period == "annual" and self.time_step == "weekly":
                        lam = rate / 52  # Direct rate scaling
                    elif period == "weekly" and self.time_step == "annual":
                        lam = rate * 52
                    else:
                        raise ValueError(f"Cannot convert {period} → {self.time_step}")

                    lam_dict[to_state] = max(lam, 0)  # Ensure non-negative

                total_lam = sum(lam_dict.values())

                if np.isclose(total_lam, 0.0):
                    probs[idx] += remaining_prob  # Allocate remaining to self
                else:
                    survival = np.exp(
                        -total_lam
                    )  # Self-transition probability  P(T>t)=e**(−λt), t=1
                    for to_state, lam in lam_dict.items():
                        to_idx = self.state_indices[to_state]
                        #  Conditional probability of transitioning to to_state (Normalizing factor)  * Cumulative distribution function
                        #  The normalizing factor distributes the total transition probability among the possible destination states based on their individual rates
                        #  and ensures the total sum of probabilities does not exceed 1 even with given direct probabilities as it distributes to remaining probabilities
                        probs[to_idx] += (
                            (lam / total_lam) * remaining_prob * (1 - survival)
                            if total_lam > 0
                            else 0
                        )
                    probs[idx] += remaining_prob * survival

            elif not rate_transitions:
                # No rate-based transitions, allocate remaining to self
                probs[idx] += remaining_prob

            # --- Validation ---
            if not np.allclose(sum(probs), 1.0, rtol=1e-5):
                raise ValueError(
                    f"Probabilities for {state} do not sum to 1 (got {sum(probs):.6f})"
                )

            matrix[idx] = probs
        return matrix

    # Deprecated
    def build_restricted(self) -> List[List[float]]:
        """
        Build the transition probability matrix for the Markov model.

        Handles:
            - Direct probabilities
            - Competing risks from event rates
            - Special transitions (explicit rows)

        Returns:
            Transition matrix as a list of lists
        """
        n = len(self.states)
        matrix = [[0.0] * n for _ in range(n)]

        # --- Handle special transitions directly ---
        for state, probs in self.special_transitions.items():
            idx = self.state_indices[state]
            matrix[idx] = probs[:]  # copy to avoid mutation

        # --- Handle regular states ---
        for state in self.states:
            if state in self.special_transitions:
                continue  # already assigned

            idx = self.state_indices[state]
            probs = [0.0] * n

            # Collect outgoing transitions
            transitions = {
                to_state: (value, period)
                for (from_state, to_state), (
                    value,
                    period,
                ) in self.transition_pairs.items()
                if from_state == state
            }

            if not transitions:
                # No outgoing transitions → absorbing/self-loop
                probs[idx] = 1.0
                matrix[idx] = probs
                continue

            # Check for type of transitions
            periods = {period for _, period in transitions.values()}
            has_none = None in periods
            has_rates = len(periods - {None}) > 0

            if has_none and has_rates:
                raise ValueError(
                    f"Cannot mix direct probabilities and rates in {state}"
                )

            # --- Case 1: Direct probabilities ---
            if has_none:
                explicit_probs = {}
                for to_state, (value, period) in transitions.items():
                    p = self.get_probability(value, period)  # type: ignore # passthrough if period=None
                    explicit_probs[to_state] = p

                total_explicit = sum(explicit_probs.values())

                # Assign
                for to_state, p in explicit_probs.items():
                    probs[self.state_indices[to_state]] = p

                if state not in explicit_probs:
                    # allocate leftover to self
                    probs[idx] = max(0.0, 1.0 - total_explicit)

                # Normalize if overshooting
                if total_explicit > 1.0:
                    for j in range(n):
                        probs[j] /= total_explicit

            # --- Case 2: Rate-based competing risks ---
            else:
                lam_dict = {}
                for to_state, (value, period) in transitions.items():
                    rate = float(value)

                    # hazard-based conversion (exact)
                    if period == self.time_step:
                        lam = -np.log(1 - prob_at_least_one(rate))
                    elif period == "annual" and self.time_step == "weekly":
                        lam = -np.log(1 - prob_at_least_one(rate)) / 52
                    elif period == "weekly" and self.time_step == "annual":
                        lam = -np.log(1 - prob_at_least_one(rate)) * 52
                    else:
                        raise ValueError(f"Cannot convert {period} → {self.time_step}")

                    lam_dict[to_state] = lam

                total_lam = sum(lam_dict.values())

                if np.isclose(total_lam, 0.0):
                    # No hazard → self-loop
                    probs[idx] = 1.0
                else:
                    survival = np.exp(-total_lam)  # stay in same state
                    for to_state, lam in lam_dict.items():
                        to_idx = self.state_indices[to_state]
                        probs[to_idx] = (lam / total_lam) * (1 - survival)

                    # Always add survival to self
                    probs[idx] += survival

            # --- Validation ---
            if not np.allclose(sum(probs), 1.0, rtol=1e-5):
                raise ValueError(
                    f"Probabilities for {state} do not sum to 1 (got {sum(probs):.6f})"
                )

            matrix[idx] = probs

        return matrix
