from typing import List, Union, Generator, Optional, Callable, Dict, Literal, Tuple
from deprecated import deprecated
from model import constants
from model.utils import cal_body_weight
from model.utils import probability_at_least_one_event
from SALib.sample import saltelli
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import multiprocessing
import enlighten
import math


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MarkovChains:
    """A Markov chain implementation that generates state transitions with reward tracking and dynamic chain switching."""

    def __init__(
        self,
        chains: Dict[str, tuple[List[str], Union[List[List[float]], np.ndarray]]],
        start_state: str,
        start_chain: str,
        steps: int,
        switch_conditions: Optional[Dict[str, Callable]] = None,
        **psa_kwargs,
    ) -> None:
        """
        Initialize Markov chain with multiple chains, transitions, and optional reward function.

        Args:
            chains: Dictionary of chain names to (states, transition_matrix) tuples
            start_state: Initial state for the chain
            start_chain: Name of the initial chain to use
            steps: Number of steps to simulate
            switch_conditions: Dictionary of chain names to switch condition functions
            psa_kwargs: Additional keyword arguments for reward and switch functions
        """
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
            bleeds_count = calculate_number_of_bleeds(
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
                bleeds_count = calculate_number_of_bleeds(
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
            Tuple[str, str], Tuple[Union[float, str], Literal["weekly", "annual"]]
        ],
        special_transitions: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        """
        Initialize ProbabilityBuilder with states and transition pairs.

        Args:
            states: List of states for the transition matrix.
            transition_pairs: Dictionary of (from_state, to_state) -> (value, period) pairs.
                              Value is a probability or rate; period is 'annual', 'weekly', or None (for direct probabilities).
            special_transitions: Optional dictionary of state -> transition probabilities for states with fixed transitions.
        """
        self.states = states
        self.transition_pairs = transition_pairs
        self.special_transitions = special_transitions or {}
        self.state_indices = {state: idx for idx, state in enumerate(states)}

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
        self, value: Union[float, str], period: Literal["weekly", "annual"]
    ) -> float:
        """
        Convert a rate or probability to a probability value.

        Note:
            if period value is None, use passed value directly as probability

        Args:
            value: The probability or rate.
            period: The period for rate conversion ('annual', 'weekly', or None).

        Returns:
            The calculated probability.
        """
        if period is None:
            return float(value)
        return probability_at_least_one_event(float(value), period)

    def get_matrix(self) -> List[List[float]]:
        """
        Construct transition matrix based on provided states and transition pairs.

        Returns:
            Transition matrix as a list of lists.
        """
        n = len(self.states)
        matrix = [[0.0] * n for _ in range(n)]

        # Handle special transitions (e.g., Death, LT_Bleeding)
        for state, probs in self.special_transitions.items():
            idx = self.state_indices[state]
            matrix[idx] = probs

        # Handle regular transitions
        for state in self.states:
            idx = self.state_indices[state]
            if state in self.special_transitions:
                continue  # Skip states with special transitions

            # Collect probabilities for transitions from this state
            probs = [0.0] * n
            total_prob = 0.0
            for (from_state, to_state), (
                value,
                period,
            ) in self.transition_pairs.items():
                if from_state == state:
                    to_idx = self.state_indices[to_state]
                    prob = self.get_probability(value, period)
                    probs[to_idx] = prob
                    total_prob += prob

            # Normalize to ensure row sums to 1
            if total_prob > 1:
                print(
                    f"Warning: Sum of probabilities ({total_prob}) for state {state} exceeds 1, normalizing"
                )
                for i in range(n):
                    probs[i] /= total_prob
            elif total_prob < 1:
                # Assign remaining probability to staying in the same state
                probs[idx] = 1 - total_prob

            matrix[idx] = probs

        return matrix


def worker_function(
    abrs: np.ndarray, kwargs: dict, strategy: Literal["on_demand", "prophylaxis"]
):
    """
    Generic worker function for both on_demand and prophylaxis treatment strategy.

    Args:
        abrs: Input array of ABR values (multiple samples per iteration)
        kwargs: MarkovChains keyword arguments
        strategy: "on_demand" or "prophylaxis"

    Returns:
        List of (input_dict, result_dict) tuples, one per ABR value.
    """
    results = []

    for abr_value in abrs:
        abr = round(abr_value)
        ajbr = round(0.75 * abr)
        ltb_fraction = 0.045  # 4.5 percent of overall bleed events
        ltb = abr * ltb_fraction
        no_bleeding_weeks = 52 - abr

        chains = kwargs["chains"]
        primary_states = chains["primary"][0]
        secondary_states = chains["secondary"][0]

        # ---- Transition pairs (shared across both) ----
        primary_transition_pairs = {
            ("Healthy", "Bleeding"): (abr, "annual"),
            ("Healthy", "Hemarthrosis"): ((abr - ajbr), "annual"),
            ("Healthy", "LT_Bleeding"): (ltb, "annual"),
            ("Bleeding", "Healthy"): (no_bleeding_weeks, "annual"),
            ("Bleeding", "Death"): (0, None),
            ("Hemarthrosis", "Healthy"): (no_bleeding_weeks, "annual"),
            ("Hemarthrosis", "Arthropathy"): (0.0015, None),  # Placeholder
            ("Arthropathy", "Healthy"): (0, None),
            ("Arthropathy", "Death"): (0, None),
            ("LT_Bleeding", "Death"): (0.2, None),
        }

        secondary_transition_pairs = {
            ("Healthy", "Bleeding"): ((abr - ajbr) / 52, "weekly"),
            ("Healthy", "Hemarthrosis"): (ajbr / 52, "weekly"),
            ("Bleeding", "Healthy"): (no_bleeding_weeks, "annual"),
            ("Bleeding", "Death"): (0, None),
            ("Hemarthrosis", "Healthy"): (no_bleeding_weeks, "annual"),
            ("LT_Bleeding", "Death"): (0.2, None),
        }

        # Primary states: ["Healthy", "Bleeding", "Hemarthrosis", "Arthropathy", "LT_Bleeding", "Death"]
        primary_special_transitions = {
            "Death": [0.0] * (len(primary_states) - 1) + [1.0],  # Absorbing
            "LT_Bleeding": [0.8] + [0.0] * (len(primary_states) - 2) + [0.2],
        }
        # Secondary_states = ["Healthy", "Bleeding", "Hemarthrosis", "LT_Bleeding", "Death"]
        secondary_special_transitions = {
            "Death": [0.0] * (len(secondary_states) - 1) + [1.0],  # Absorbing
            "LT_Bleeding": [0.8] + [0.0] * (len(secondary_states) - 2) + [0.2],
        }

        # ---- Build transition matrices ----
        primary_builder = TransitionGenerator(
            states=primary_states,
            transition_pairs=primary_transition_pairs,
            special_transitions=primary_special_transitions,
        )
        secondary_builder = TransitionGenerator(
            states=secondary_states,
            transition_pairs=secondary_transition_pairs,
            special_transitions=secondary_special_transitions,
        )
        chains["primary"] = (primary_states, primary_builder.get_matrix())
        chains["secondary"] = (secondary_states, secondary_builder.get_matrix())

        # ---- Markov model setup ----
        markov = MarkovChains(
            chains=chains,
            lambda_bleeding=(abr - ajbr) / 52,
            lambda_joint_bleeding=ajbr / 52,
            **{
                k: v for k, v in kwargs.items() if k != "chains"
            },  # avoid mutating kwargs
        )

        # Select reward functions based on treatment
        if strategy == "on_demand":
            factor_func = on_demand_factor_consumption
        elif strategy == "prophylaxis":
            factor_func = prophylaxis_factor_consumption
        else:
            raise ValueError(f"Invalid treatment: {strategy}")

        markov.add_reward_function(factor_func)
        markov.add_reward_function(construct_utility_reward_function(strategy))
        markov.run()
        rewards = markov.collect_rewards()

        # ---- Results ----
        n_cycles = kwargs.get("steps")
        if not n_cycles or not isinstance(n_cycles, int):
            raise ValueError("Model number of steps not correctly defined for psa.")

        input_dict = {"abr": abr, "ajbr": ajbr}
        result_dict = {}

        total_factor_use = np.sum(rewards[factor_func.__name__][1:])
        total_utility_values = np.sum(rewards["utility_reward_function"][1:])
        factor_consumption_list: list = rewards[factor_func.__name__][1:]

        # Costs
        if constants.DISCOUNT_RATE_WEEKLY:
            factor_costs = [
                (
                    dose
                    * constants.PRICE_PER_UI_FACTOR_VIII
                    / constants.PPP_CONVERSION_FACTOR
                )
                / (1 + constants.DISCOUNT_RATE_WEEKLY) ** i
                for i, dose in enumerate(factor_consumption_list)
            ]
        else:
            factor_costs = [
                dose * constants.PRICE_PER_UI_FACTOR_VIII / constants.RIAL_USD_PRICE
                for dose in factor_consumption_list
            ]

        # Store aggregated results
        result_dict["total_factors_use"] = total_factor_use
        result_dict["total_factors_costs"] = np.sum(factor_costs) / (n_cycles / 52)
        result_dict["annual_factor_consumption"] = total_factor_use / (n_cycles / 52)
        result_dict["QALYS"] = total_utility_values

        results.append((input_dict, result_dict))

    return results


def psa_simulation(strategy: str, n_samples: int, **kwargs):
    """
    Run PSA simulation for a given treatment strategy.

    Args:
        strategy: Either "on_demand" or "prophylaxis"
        n_samples: Number of samples
        **kwargs: MarkovChain keyword arguments (except transitions)

    Returns:
        Tuple of (inputs, results) containing simulation inputs and outputs
    """
    # Define ABR bounds depending on strategy
    abr_bounds = {
        "on_demand": [0, 44],
        "prophylaxis": [0, 28],
    }
    if strategy not in abr_bounds:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Define the problem for sampling
    problem = {"num_vars": 1, "names": ["ABR"], "bounds": [abr_bounds[strategy]]}
    param_samples = saltelli.sample(problem, n_samples, calc_second_order=False)

    inputs = []
    results = {
        "total_factors_use": [],
        "total_factors_costs": [],
        "annual_factor_consumption": [],
        "QALYS": [],
    }

    manager = enlighten.get_manager()
    progress_bar: enlighten.Counter = manager.counter(
        total=len(param_samples), desc=f"Simulating {strategy}:", unit="simulation"
    )

    def update_bar(_):
        progress_bar.update(incr=1)

    with multiprocessing.Pool(processes=6) as pool:
        async_results = [
            pool.apply_async(
                worker_function, args=(abr, kwargs, strategy), callback=update_bar
            )
            for abr in param_samples
        ]
        # Collect results
        for res in async_results:
            for input_dict, result_dict in res.get():
                inputs.append(input_dict)
                results["total_factors_use"].append(result_dict["total_factors_use"])
                results["total_factors_costs"].append(
                    result_dict["total_factors_costs"]
                )
                results["annual_factor_consumption"].append(
                    result_dict["annual_factor_consumption"]
                )
                results["QALYS"].append(result_dict["QALYS"])

    manager.stop()
    return inputs, results


def calculate_number_of_bleeds(state: str, **kwargs) -> int:
    # Conditional probability formula
    def conditional_probs(k: int, l: float):
        return (l**k) / (math.factorial(k) * (math.exp(l) - 1))

    # Get lambda_value from kwargs
    lambda_value = (
        kwargs.get("lambda_bleeding")
        if state.lower() == "bleeding"
        else (
            kwargs.get("lambda_joint_bleeding")
            if state.lower() == "joint_bleeding"
            else 0
        )
    )
    number_of_bleeds = 1
    if lambda_value != 0:
        if not isinstance(lambda_value, float):
            raise TypeError("No valid lambda value provided.")
        events_probs = [conditional_probs(k, lambda_value) for k in range(1, 5, 1)]
        # Normalizing
        events_probs = [p / sum(events_probs) for p in events_probs]
        number_of_bleeds = np.random.choice([i for i in range(1, 5, 1)], p=events_probs)
    return number_of_bleeds


def on_demand_factor_consumption(
    step: int, state: str, number_of_bleeds: int, **psa_kwargs
):
    """Reward function that calculates factor viii consumption per bleed even per patient body weight over time."""
    # starting treatment from age 2 yo
    weight = cal_body_weight(step, b=2 * 52)
    injected_dose = 0
    if state.lower() == "bleeding":
        # Muscle | illopsoas | Renal | Oral mucosa and dental
        injected_dose = round(weight * constants.BLEEDING_DOSE) * number_of_bleeds
    elif state.lower() == "joint_bleeding":
        # Joint
        injected_dose = round(weight * constants.JOINT_BLEEDING_DOSE) * number_of_bleeds
    elif state.lower() == "lt_bleeding":
        # Intra_cranial | Gastro | Neck & throat
        injected_dose = round(weight * constants.LT_BLEEDING_DOSE)
    return injected_dose


def prophylaxis_factor_consumption(
    step: int, state: str, number_of_bleeds: int, **psa_kwargs
) -> int:
    """
    Reward function that calculates factor viii consumption per bleed even per patient body weight over time.
    """
    # starting treatment from age 2 yo
    weight = cal_body_weight(step, b=2 * 52)
    injected_dose = round(weight * constants.STANDARD_PROPHYLAXIS_WEEKLY_DOSE)
    if state.lower() == "bleeding":
        # Muscle | illopsoas | Renal | Oral mucosa and dental
        injected_dose += round(weight * constants.BLEEDING_DOSE) * number_of_bleeds
    elif state.lower() == "joint_bleeding":
        # Joint
        injected_dose += (
            round(weight * constants.JOINT_BLEEDING_DOSE) * number_of_bleeds
        )
    elif state.lower() == "lt_bleeding":
        # Intra_cranial | Gastro | Neck & throat
        injected_dose += round(weight * constants.LT_BLEEDING_DOSE)
    return injected_dose


def construct_utility_reward_function(
    treatment: Literal["on_demand", "prophylaxis"], decrement_utility: bool = False
) -> Callable[[int, str], float]:
    """
    Factory function to create a utility reward function.
    Tracks bleeds and applies discounting or utility decrement
    """
    bleeds_count = [0]  # closure mutable container

    # --- Configuration ---
    base_utilities = {
        "on_demand": {"healthy": 0.85},
        "prophylaxis": {"healthy": 0.915},
    }

    # Default (non-healthy) utilities
    state_utilities = {
        "arthropathy": 0.75,
        "bleeding": 0.60,
        "joint_bleeding": 0.50,
        "lt_bleeding": 0.25,
        "death": 0.0,
    }

    decrement_per_bleed = {
        "on_demand": 0.0003725,
        "prophylaxis": 0.0018,
    }[treatment]

    def discount(value: float, step: int) -> float:
        """Apply weekly discounting if enabled."""
        if constants.DISCOUNT_RATE_WEEKLY:
            return value / (1 + constants.DISCOUNT_RATE_WEEKLY) ** step
        return value

    def utility_reward_function(step: int, state: str, **kwargs) -> float:
        """Returns utility values for each health state with event-based decay for healthy state."""
        nonlocal bleeds_count
        state_lower = state.lower()

        # Increment cumulative bleeds based on the state at this step
        if state_lower in ["bleeding", "joint_bleeding"]:
            bleeds_count[0] += 1

        if state_lower == "healthy":
            base_u = base_utilities[treatment]["healthy"]
            decrement = (
                decrement_per_bleed * bleeds_count[0] if decrement_utility else 0
            )
            adjusted_u = max(0.65, base_u - decrement)
            utility = adjusted_u
        else:
            utility = state_utilities.get(state_lower, 0.0)

        # Normalize to per-week utility and apply discount
        return discount(utility * (1 / 52), step)

    return utility_reward_function


def visualize_transition_matrix(
    matrix: np.ndarray, states: list, title: str = "Transition Matrix"
):
    """
    Visualize a transition matrix as a heatmap.

    Args:
        matrix: Transition probability matrix (square np.ndarray).
        states: List of state names (same length as matrix).
        title: Title for the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        xticklabels=states,
        yticklabels=states,
        cmap="Blues",
    )
    plt.title(title)
    plt.xlabel("Next State")
    plt.ylabel("Current State")
    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300, bbox_inches="tight")
