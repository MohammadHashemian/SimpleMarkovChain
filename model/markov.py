from typing import List, Union, Generator, Optional, Callable, Dict, Literal, Tuple
from dataclasses import dataclass
from model import constants
from model.utils import cal_body_weight
from model.utils import probability_at_least_one_event
from SALib.sample import saltelli
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import multiprocessing
import enlighten
import math


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class MarkovResults:
    sequences: list
    total_factor_use: float
    total_factor_costs: float
    annual_factor_consumption: float
    annual_factor_costs: float
    qaly: float


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
    abr: float, kwargs: dict, strategy: Literal["on_demand", "prophylaxis"]
):
    """
    Worker function for simulating Markov chains for a given ABR and strategy.

    Args:
        abr: Annual bleeding rate (ABR).
        kwargs: MarkovChains keyword arguments.
        strategy: "on_demand" or "prophylaxis".

    Returns:
        Tuple of (input_dict, result_dict) containing simulation inputs and outputs.
    """
    abr = round(abr)
    weekly_bleeding_rate = abr / 52.0
    weekly_ajbr_rate = (
        constants.AJBR_FRACTION * abr
    ) / 52.0  # Weekly joint bleeding rate
    weekly_aebr_rate = (
        weekly_bleeding_rate - weekly_ajbr_rate
    )  # Weekly non-joint bleeding rate
    weekly_ltb_rate = (
        constants.LTB_FRACTION * abr
    ) / 52.0  # Weekly life-threatening bleeding rate

    # Ensure non-negative probability for staying in Healthy state
    no_bleeding_weeks_pro = max(
        0, 1 - weekly_aebr_rate - weekly_ajbr_rate - weekly_ltb_rate
    )

    chains = kwargs["chains"]
    primary_states = chains["primary"][0]
    secondary_states = chains["secondary"][0]

    # ---- Transition pairs (shared across both) ----
    primary_transition_pairs = {
        ("Healthy", "Bleeding"): (weekly_aebr_rate, "weekly"),
        ("Healthy", "Hemarthrosis"): (weekly_ajbr_rate, "weekly"),
        ("Healthy", "LT_Bleeding"): (weekly_ltb_rate, "weekly"),
        ("Bleeding", "Healthy"): (no_bleeding_weeks_pro, None),
        ("Bleeding", "LT_Bleeding"): (weekly_ltb_rate, "weekly"),
        ("Bleeding", "Death"): (0, None),
        ("Hemarthrosis", "Healthy"): (no_bleeding_weeks_pro, None),
        ("Hemarthrosis", "LT_Bleeding"): (weekly_ltb_rate, "weekly"),
        ("Hemarthrosis", "Arthropathy"): (
            constants.HEMARTHROSIS_TO_ARTHROPATHY,
            None,
        ),
        ("Arthropathy", "Healthy"): (0, None),
        ("Arthropathy", "Death"): (0, None),
    }

    secondary_transition_pairs = {
        ("Healthy", "Bleeding"): (weekly_aebr_rate, "weekly"),
        ("Healthy", "Hemarthrosis"): (weekly_ajbr_rate, "weekly"),
        ("Healthy", "LT_Bleeding"): (weekly_ltb_rate, "weekly"),
        ("Bleeding", "Healthy"): (no_bleeding_weeks_pro, None),
        ("Bleeding", "LT_Bleeding"): (weekly_ltb_rate, "weekly"),
        ("Bleeding", "Death"): (0, None),
        ("Hemarthrosis", "Healthy"): (no_bleeding_weeks_pro, None),
        ("Hemarthrosis", "LT_Bleeding"): (weekly_ltb_rate, "weekly"),
    }

    # Primary states: ["Healthy", "Bleeding", "Hemarthrosis", "Arthropathy", "LT_Bleeding", "Death"]
    primary_special_transitions = {
        "Death": [0.0] * (len(primary_states) - 1) + [1.0],  # Absorbing
        "LT_Bleeding": [0.8] + [0.0] * (len(primary_states) - 2) + [0.2],
    }
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
        lambda_bleeding=weekly_aebr_rate,  # Corrected: Weekly non-joint bleeding rate
        lambda_joint_bleeding=weekly_ajbr_rate,  # Corrected: Weekly joint bleeding rate
        **{k: v for k, v in kwargs.items() if k != "chains"},
    )

    # Select reward functions based on treatment
    if strategy == "on_demand":
        factor_func = on_demand_factor_consumption_wrapper
    elif strategy == "prophylaxis":
        factor_func = prophylaxis_factor_consumption_wrapper
    else:
        raise ValueError(f"Invalid treatment: {strategy}")

    markov.add_reward_function(factor_func)
    markov.add_reward_function(construct_utility_reward_function(strategy))
    sequences = markov.run()
    rewards = markov.collect_rewards()

    # ---- Results ----
    n_cycles = kwargs.get("steps")
    if not n_cycles or not isinstance(n_cycles, int):
        raise ValueError("Model number of steps not correctly defined for psa.")

    inputs = {"abr": abr, "ajbr": weekly_ajbr_rate * 52, "chains": chains}

    _utility_func_name = construct_utility_reward_function.__name__.removeprefix(
        "construct_"
    )
    total_factor_use = np.sum(rewards[factor_func.__name__][1:])
    total_utility_values = np.sum(rewards[_utility_func_name][1:])
    factor_consumption_list: list = rewards[factor_func.__name__][1:]
    # Costs (corrected for consistent discounting and currency conversion)
    factor_costs = [
        (
            (
                dose
                * constants.PRICE_PER_UI_FACTOR_VIII
                / constants.PPP_CONVERSION_FACTOR  # Use PPP for consistency
            )
            / (1 + constants.DISCOUNT_RATE_WEEKLY) ** i
            if constants.DISCOUNT_RATE_WEEKLY
            else dose
            * constants.PRICE_PER_UI_FACTOR_VIII
            / constants.PPP_CONVERSION_FACTOR
        )
        for i, dose in enumerate(factor_consumption_list)
    ]

    # Store aggregated results
    res = MarkovResults(
        total_factor_use=total_factor_use,
        total_factor_costs=np.sum(factor_costs),
        annual_factor_consumption=total_factor_use / (n_cycles / 52.0),
        annual_factor_costs=np.sum(factor_costs) / (n_cycles / 52.0),
        qaly=total_utility_values,
        sequences=sequences,
    )
    return (inputs, res)


def psa_simulation(strategy: str, n_samples: int, **kwargs) -> tuple[list, list[MarkovResults]]:
    """
    Run PSA simulation for a given treatment strategy.

    Args:
        strategy: Either "on_demand" or "prophylaxis".
        n_samples: Number of samples.
        **kwargs: MarkovChain keyword arguments (except transitions).

    Returns:
        Tuple of (inputs, results) containing simulation inputs and outputs.
    """
    # Study data for ABR (Mean, SD)
    on_demand_abr = np.array(
        [
            [58.3, 26.9],  # Zhao et al.
            [37.2, 19.9],  # Manco-Johnson MJ et al.
            [19.5, 15.0],  # Tagliaferri A et al.
            [17.7, 11.7],  # Tagliaferri A et al.
            [13, 0],  # Gringeri A et al.
            [7.4, 9.5],  # Romanová G et al.
            [13.2, 12.43],  # Berntorp E et al.
            [14.0, 12.3],  # Khair K et al.
            [18.4, 14.2],  # Khair K et al.
            [15.8, 8.13],  # Khair K et al.
        ]
    )
    prophylaxis_abr = np.array(
        [
            [2.5, 4.6],  # Zhao et al.
            [2.5, 4.7],  # Manco-Johnson MJ et al.
            [2.6, 2.2],  # Tagliaferri A et al.
            [4.5, 7.1],  # Tagliaferri A et al.
            [6.3, 0],  # Gringeri A et al.
            [2.1, 2.1],  # Romanová G et al.
            [4.26, 5.97],  # Berntorp E et al.
            [3.5, 4.3],  # Khair K et al.
            [3.3, 4.1],  # Khair K et al.
            [3.7, 3.9],  # Khair K et al.
        ]
    )
    study_data = on_demand_abr if strategy == "on_demand" else prophylaxis_abr
    num_studies = study_data.shape[0]

    problem = {
        "num_vars": num_studies,
        "names": [f"study_{i}_weight" for i in range(num_studies)],
        "bounds": [[0, 1]] * num_studies,
    }
    param_samples = saltelli.sample(problem, n_samples, calc_second_order=True)

    def weights_to_abr(weights):
        """Convert study weights to an ABR value using Gamma distribution sampling."""
        weights = np.array(weights)
        probabilities = np.exp(weights) / np.sum(np.exp(weights))
        chosen_study_idx = np.random.choice(range(num_studies), p=probabilities)
        mu, sigma = study_data[chosen_study_idx]

        if sigma <= 0 or np.isnan(sigma):
            return max(0, mu)  # Ensure non-negative ABR

        k = (mu / sigma) ** 2
        theta = (sigma**2) / mu

        if k <= 0 or theta <= 0 or np.isnan(k) or np.isnan(theta):
            return max(0, mu)

        return max(0, np.random.gamma(k, theta))

    # ---- Debug ----
    abr_values = np.array([weights_to_abr(params) for params in param_samples])
    visualize_abr(abr_values, strategy)

    inputs = []
    results = []

    manager = enlighten.get_manager()
    progress_bar = manager.counter(
        total=len(param_samples), desc=f"Simulating {strategy}:", unit="simulation"
    )

    def update_bar(_):
        progress_bar.update(incr=1)

    with multiprocessing.Pool(processes=6) as pool:
        async_results = [
            pool.apply_async(
                worker_function, args=(abr, kwargs, strategy), callback=update_bar
            )
            for abr in abr_values
        ]
        for res in async_results:
            input_dict, res = res.get()
            inputs.append(input_dict)
            results.append(res)
    manager.stop()
    return inputs, results


def factor_consumption(
    step: int,
    state: str,
    number_of_bleeds: int,
    strategy: Literal["on_demand", "prophylaxis"],
    **psa_kwargs,
) -> int:
    """
    Reward function that calculates factor VIII consumption per patient body weight over time.

    Args:
        step (int): Current Markov cycle step (weeks).
        state (str): Current patient state (e.g., bleeding, joint_bleeding, lt_bleeding).
        number_of_bleeds (int): Number of bleeding events in this step.
        mode (str): Either 'on_demand' or 'prophylaxis'.

    Returns:
        int: Total factor VIII consumption (IU).
    """
    # starting treatment from age 2 yo
    weight = cal_body_weight(step, b=constants.START_SIMULATION_AGE_IN_WEEK)
    injected_dose = 0

    # Add prophylaxis baseline if applicable
    injected_dose += (
        round(weight * constants.STANDARD_PROPHYLAXIS_WEEKLY_DOSE)
        if strategy.lower() == "prophylaxis"
        else 0
    )

    # Bleeding-related consumption
    match state.lower():
        case "bleeding":
            injected_dose += round(weight * constants.BLEEDING_DOSE) * number_of_bleeds
        case "joint_bleeding":
            injected_dose += (
                round(weight * constants.JOINT_BLEEDING_DOSE) * number_of_bleeds
            )
        case "lt_bleeding":
            injected_dose += round(weight * constants.LT_BLEEDING_DOSE)

    return injected_dose


def on_demand_factor_consumption_wrapper(
    step: int, state: str, number_of_bleeds: int, **psa_kwargs
):
    return factor_consumption(
        step, state, number_of_bleeds, strategy="on_demand", **psa_kwargs
    )


def prophylaxis_factor_consumption_wrapper(
    step: int, state: str, number_of_bleeds: int, **psa_kwargs
) -> int:
    return factor_consumption(
        step, state, number_of_bleeds, strategy="prophylaxis", **psa_kwargs
    )


def construct_utility_reward_function(
    treatment: Literal["on_demand", "prophylaxis"], decrement_utility: bool = False
) -> Callable[[int, str, int], float]:
    """
    Factory function to create a utility reward function.
    Tracks bleeds and applies discounting or utility decrement
    """
    bleeds_count = [0]  # closure mutable container

    # --- Configuration ---
    # NOTE
    # same or different base utility? 0.85, 0.915
    base_utilities = {
        "on_demand": {"healthy": 0.915},
        "prophylaxis": {"healthy": 0.915},
    }

    # Default (non-healthy) utilities
    state_utilities = {
        "arthropathy": 0.75,
        "bleeding": 0.60,
        "hemarthrosis": 0.50,
        "lt_bleeding": 0.25,
        "death": 0.0,
    }

    # Placeholder values
    decrement_per_bleed = {
        "on_demand": 0.0003725,
        "prophylaxis": 0.0018,
    }[treatment]

    def discount(value: float, step: int) -> float:
        """Apply weekly discounting if enabled."""
        if constants.DISCOUNT_RATE_WEEKLY:
            return value / (1 + constants.DISCOUNT_RATE_WEEKLY) ** step
        return value

    def utility_reward_function(
        step: int, state: str, number_of_bleeds: int, **kwargs
    ) -> float:
        """Returns utility values for each health state with event-based decay for healthy state."""
        nonlocal bleeds_count
        state_lower = state.lower()

        # Increment cumulative bleeds based on the state at this step
        if state_lower in ["bleeding", "hemarthrosis"]:
            bleeds_count[0] += number_of_bleeds

        if state_lower == "healthy":
            base_u = base_utilities[treatment]["healthy"]
            decrement = (
                decrement_per_bleed * bleeds_count[0] if decrement_utility else 0
            )
            adjusted_u = max(0.65, base_u - decrement) if decrement_utility else base_u
            utility = adjusted_u
        else:
            utility = state_utilities.get(state_lower, 0.0)

        # Normalize to per-week utility and apply discount
        return discount(utility * (1 / 52), step)

    return utility_reward_function


def count_bleeds_conditional_prob(state: str, **kwargs) -> int:
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


def count_bleeds_poisson(state: str, **kwargs) -> int:
    """
    Calculate the number of bleeding events in a given state using a Poisson distribution.

    Args:
        state (str): Current patient state (e.g., 'Bleeding', 'Joint_Bleeding').
        **kwargs: Additional parameters, including 'lambda_bleeding' and 'lambda_joint_bleeding'.

    Returns:
        int: Number of bleeding events.
    """
    # Get lambda value based on state
    lambda_value = (
        kwargs.get("lambda_bleeding")
        if state.lower() == "bleeding"
        else (
            kwargs.get("lambda_joint_bleeding")
            if state.lower() == "joint_bleeding"
            else 0
        )
    )

    if (
        lambda_value is None
        or not isinstance(lambda_value, (int, float))
        or lambda_value < 0
    ):
        raise ValueError(f"Invalid lambda value for state {state}: {lambda_value}")

    if lambda_value == 0:
        return 0
    return max(1, np.random.poisson(lambda_value))


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
        fmt=".4f",
        xticklabels=states,
        yticklabels=states,
        cmap="Blues",
    )
    plt.title(title)
    plt.xlabel("Next State")
    plt.ylabel("Current State")
    plt.tight_layout()
    plt.savefig(
        PROJECT_ROOT / "outputs" / "figures" / "transitions" / f"{title}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def visualize_abr(abr_values, strategy: str):
    """
    Visualize sampled abr_values and draw histogram of it

    Args:
        abr_values: sampled abr values
        strategy: on_demand or prophylaxis

    Returns:
        None: stores figures at output directory
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(
        abr_values,
        bins=30,
        kde=True,
        color="blue" if strategy == "on_demand" else "green",
    )
    plt.title(f"ABR Distribution for {strategy.capitalize()} Strategy")
    plt.xlabel("Annual Bleeding Rate (ABR)")
    plt.ylabel("Frequency")
    plt.savefig(
        PROJECT_ROOT / "outputs" / "figures" / f"abr_distribution_{strategy}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
