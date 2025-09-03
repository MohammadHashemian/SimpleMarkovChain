from typing import List, Union, Generator, Optional, Callable, Dict, Literal, Tuple
from dataclasses import dataclass
from model.utils import count_bleeds_poisson, prob_at_least_one
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Results:
    """
    Attrs:
        sequences: Markov chain simulated steps results
        total_factor_use: Sum of rewards for factor viii consumption
        total_factor_costs: Sum of rewards for factors multiplied to it's relative cost
        annual_factor_consumption: annualized annual_factor_consumption
        annual_Factor_costs: annualized total_factor_costs
        qaly: Sum of qalys over simulation periods
    """

    sequences: list
    total_factor_use: float
    total_factor_costs: float
    annual_factor_consumption: float
    annual_factor_costs: float
    qaly: float


class MarkovChains:
    """
    Summary
    --------
    A Markov chain implementation that generates state transitions with reward tracking and dynamic chain switching.

    Initialize Markov chain with multiple chains, transitions, and optional reward function.

    Args:
        chains: Dictionary of chain names to (states, transition_matrix) tuples
        start_state: Initial state for the chain
        start_chain: Name of the initial chain to use
        steps: Number of steps to simulate
        switch_conditions: Dictionary of chain names to switch condition functions
        psa_kwargs: Additional keyword arguments for reward and switch functions
    """

    def __init__(
        self,
        chains: Dict[str, Tuple[List[str], Union[List[List[float]], np.ndarray]]],
        start_state: str,
        start_chain: str,
        steps: int,
        switch_conditions: Optional[Dict[str, Callable]] = None,
        **psa_kwargs,
    ) -> None:
        self.chains = {
            name: (states, np.array(transitions, dtype=np.float64))
            for name, (states, transitions) in chains.items()
        }
        self.steps = steps
        self.reward_functions: List[Callable] = []
        self.rewards: Dict[str, List[float | int]] = {}
        self.psa_kwargs = psa_kwargs
        self.switch_conditions = switch_conditions or {}
        self.current_chain = start_chain
        self.current_state_idx: int = 0

        # Validate all chains
        for chain_name, (states, transitions) in self.chains.items():
            if transitions.shape != (len(states), len(states)):
                raise ValueError(
                    f"Chain '{chain_name}': Expected {len(states)}x{len(states)} transition matrix, got {transitions.shape}"
                )
            if not np.allclose(transitions.sum(axis=1), 1, rtol=1e-5):
                raise ValueError(
                    f"Chain '{chain_name}': Each row in the transition matrix must sum to 1"
                )
            if chain_name == start_chain and start_state not in states:
                raise ValueError(
                    f"Start state '{start_state}' not in states list of chain '{start_chain}'"
                )

        if start_chain not in self.chains:
            raise ValueError(f"Start chain '{start_chain}' not in provided chains")

        # Set initial state index
        self.current_state_idx = self.chains[start_chain][0].index(start_state)

    def add_reward_function(self, func: Callable) -> None:
        """Add a reward function to be calculated at each step."""
        self.reward_functions.append(func)
        self.rewards[func.__name__] = []

    def _get_current_chain(self) -> tuple[List[str], np.ndarray]:
        """Return active states and transitions."""
        return self.chains[self.current_chain]

    def walk(self, steps: Optional[int] = None) -> Generator[str, None, None]:
        """Generate a sequence of states for the specified number of steps."""
        if steps is not None:
            self.steps = steps

        current_state_idx: int = self.current_state_idx
        states, transitions = self._get_current_chain()
        assert isinstance(states, list), "States must be a list of strings"

        # Rewards for initial state
        if self.reward_functions:
            bleeds_count = count_bleeds_poisson(
                state=states[current_state_idx], **self.psa_kwargs
            )
            for func in self.reward_functions:
                r = func(
                    step=0,
                    state=states[current_state_idx],
                    number_of_bleeds=bleeds_count,
                    chain=self.current_chain,
                    **self.psa_kwargs,
                )
                self.rewards[func.__name__].append(r)

        for step in range(self.steps):
            yield states[current_state_idx]

            # Check for chain switch
            for chain_name, condition in self.switch_conditions.items():
                if chain_name != self.current_chain and condition(
                    step,
                    states[current_state_idx],
                    self.current_chain,
                    **self.psa_kwargs,
                ):
                    self.current_chain = chain_name
                    states, transitions = self._get_current_chain()
                    assert isinstance(states, list), "States must be a list of strings"
                    # Reset state if it doesn't exist in new chain
                    if states[current_state_idx] not in states:
                        current_state_idx = 0
                    else:
                        current_state_idx = states.index(states[current_state_idx])
                    break

            # Transition
            probs = transitions[current_state_idx]
            current_state_idx = np.random.choice(len(states), p=probs)

            # Rewards
            if self.reward_functions:
                bleeds_count = count_bleeds_poisson(
                    state=states[current_state_idx], **self.psa_kwargs
                )
                for func in self.reward_functions:
                    r = func(
                        step=step + 1,
                        state=states[current_state_idx],
                        number_of_bleeds=bleeds_count,
                        chain=self.current_chain,
                        **self.psa_kwargs,
                    )
                    self.rewards[func.__name__].append(r)

        yield states[current_state_idx]

    def collect_rewards(self) -> Dict[str, List[float | int]]:
        """Return all collected rewards for each reward function."""
        return self.rewards

    def run(self) -> List[str]:
        """Run the Markov chain and return the complete sequence of states."""
        return list(self.walk())


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

    def get_crm(self) -> List[List[float]]:
        """
        Build the transition probability matrix for the Markov model.

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
                    survival = np.exp(-total_lam)  # Self-transition probability
                    for to_state, lam in lam_dict.items():
                        to_idx = self.state_indices[to_state]
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

    def get_restricted_crm(self) -> List[List[float]]:
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
