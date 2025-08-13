from typing import List, Union, Generator, Optional, Callable, Dict, Literal
from model import constants
from model.utils import cal_body_weight
from model.utils import probability_at_least_one_event
from SALib.sample import saltelli
from pathlib import Path
import math
import numpy as np
import pandas as pd
import multiprocessing
import enlighten

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MarkovChain:
    """A Markov chain implementation that generates state transitions with reward tracking."""

    def __init__(
        self,
        states: List[str],
        transitions: Union[List[List[np.float64]], np.ndarray],
        start_state: str,
        steps: int,
        **psa_kwargs,
    ) -> None:
        """
        Initialize Markov chain with states, transitions, and optional reward function.

        Args:
            states: List of possible states
            transitions: Transition probability matrix
            start_state: Initial state
            steps: Number of steps to simulate
        """
        self.transitions = np.array(transitions, dtype=np.float64)
        self.states = states
        self.steps = steps
        self.reward_functions = []
        self.rewards: Dict[str, List[float | int]] = {}
        self.psa_kwargs = psa_kwargs

        # Validate inputs
        if self.transitions.ndim != 2:
            raise ValueError("Transition matrix must be 2D")
        if self.transitions.shape != (len(states), len(states)):
            raise ValueError(
                f"Expected {len(states)}x{len(states)} transition matrix, got {self.transitions.shape}"
            )
        if not np.allclose(self.transitions.sum(axis=1), 1, rtol=1e-5):
            raise ValueError("Each row in the transition matrix must sum to 1")
        if start_state not in states:
            raise ValueError(f"Start state '{start_state}' not in states list")

        self.current_state_idx = states.index(start_state)
        self.num_states = len(states)

    def add_reward_function(self, func: Callable) -> None:
        """Add a reward function to be calculated at each step."""
        self.reward_functions.append(func)
        self.rewards[func.__name__] = []

    def walk(self, steps: Optional[int] = None) -> Generator[str, None, None]:
        """Generate a sequence of states for the specified number of steps."""
        current_state = self.current_state_idx
        if steps:
            self.steps = steps

        # Calculate reward for initial state (S_0)
        if self.reward_functions:
            bleeds_count_per_week_factor = calculate_number_of_bleeds(
                state=self.states[current_state], **self.psa_kwargs
            )
            for func in self.reward_functions:
                r = func(
                    step=0,
                    state=self.states[current_state],
                    number_of_bleeds=bleeds_count_per_week_factor,
                    **self.psa_kwargs,
                )
                self.rewards[func.__name__].append(r)

        for step in range(self.steps):
            # Yield current state
            yield self.states[current_state]

            # Transition to next state
            probs = self.transitions[current_state]
            current_state = np.random.choice(self.num_states, p=probs)
            bleeds_count_per_week_factor = calculate_number_of_bleeds(
                state=self.states[current_state], **self.psa_kwargs
            )

            # Calculate rewards for the new state
            if self.reward_functions:
                for func in self.reward_functions:
                    r = func(
                        step=step,
                        state=self.states[current_state],
                        number_of_bleeds=bleeds_count_per_week_factor,
                        **self.psa_kwargs,
                    )
                    self.rewards[func.__name__].append(r)

        # Yield the final state
        yield self.states[current_state]

    def collect_rewards(self) -> dict:
        """Return all collected rewards for each reward function."""
        return self.rewards

    def run(self) -> List[str]:
        """Run the Markov chain and return the complete sequence of states."""
        return list(self.walk())


class ProbabilityBuilder:
    def __init__(self, abr: float, ajbr: float, annual_ltb_prob: float) -> None:
        self.abr = float(abr)
        self.ajbr = float(ajbr)
        self.aebr = float(abr) - float(ajbr)
        self.annual_ltb_prob = float(annual_ltb_prob)
        self.weekly_ltb_prob = float(annual_ltb_prob) / 52

    def to_bleeding(self):
        return probability_at_least_one_event(self.aebr, "annual")

    def to_joint_bleeding(self):
        return probability_at_least_one_event(self.ajbr, "annual")

    def to_ltb(self):
        return probability_at_least_one_event(self.weekly_ltb_prob, "weekly")

    def to_healthy(self):
        total = self.to_ltb() + self.to_bleeding() + self.to_joint_bleeding()
        if total > 1:
            print(f"Warning: Sum of probabilities ({total}) exceeds 1, clamping to 0")
        return max(0, 1 - total)

    def get_weekly_ltb(self):
        """
        Note: not a probability, annual ltb event divided to weeks count
        """
        return self.weekly_ltb_prob

    def get_matrix(self):
        # Columns [Healthy, Minor, Major, LTB, Death]
        matrix = [
            [
                self.to_healthy(),
                self.to_bleeding(),
                self.to_joint_bleeding(),
                self.to_ltb(),
                0,
            ],  # Row: Healthy
            [
                self.to_healthy(),
                self.to_bleeding(),
                self.to_joint_bleeding(),
                self.to_ltb(),
                0,
            ],  # Row: Minor
            [
                self.to_healthy(),
                self.to_bleeding(),
                self.to_joint_bleeding(),
                self.to_ltb(),
                0,
            ],  # Row: Major
            [0.8, 0, 0, 0, 0.2],  # Row: LTB
            [0, 0, 0, 0, 1],  # Row: Death
        ]
        return matrix


def load_transition_matrix(
    io: Path,
    sheet_name: str,
) -> tuple[str, list[str], np.ndarray]:
    """
    Load transition matrix from Excel file.

    Args:
        io: Path to Excel file
        sheet_name: Sheet containing transition matrix

    Returns:
        tuple(start_state, states, transition)
    """
    df = pd.read_excel(io, sheet_name=sheet_name)
    states = list(df.columns[1:-1])  # Exclude 'States' and 'SUM' columns
    start_state = states[0]
    transitions = df.drop(columns=["States", "SUM"]).to_numpy()
    return (start_state, states, transitions)


def on_demand_worker_function(abr, kwargs: dict):
    """
    Worker function to process a single abr value.
    Args:
        abr: Input array for ABR value
        kwargs: MarkovChain keyword arguments
    Returns:
        Tuple of (input_dict, result_dict) for one iteration.
    """
    abr = round(float(abr[0]))
    ajbr = round(0.75 * abr)
    builder = ProbabilityBuilder(abr=abr, ajbr=ajbr, annual_ltb_prob=0.021)
    transition = builder.get_matrix()
    markov = MarkovChain(
        transitions=transition,
        lambda_bleeding=(abr - ajbr) / 52,  # PSA_KWARGS
        lambda_joint_bleeding=ajbr / 52,  # PSA_KWARGS
        **kwargs,
    )
    # Wrap utility_reward_function to track cumulative bleeds
    markov.add_reward_function(on_demand_factor_consumption)
    markov.add_reward_function(construct_utility_reward_function("on_demand"))
    markov.run()
    rewards = markov.collect_rewards()
    # Store results
    n_cycles = kwargs.get("steps")
    if not n_cycles or not isinstance(n_cycles, int):
        raise ValueError("Model number of steps not correctly defined for psa.")
    input_dict = {"abr": abr, "ajbr": ajbr}
    result_dict = {}
    total_factor_use = np.sum(rewards[on_demand_factor_consumption.__name__][1:])
    total_utility_values = np.sum(rewards["utility_reward_function"][1:])
    result_dict["total_factors_use"] = total_factor_use
    factor_consumption_list: list = rewards[on_demand_factor_consumption.__name__][1:]
    # Discounted costs
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
        # Not discounted costs
        factor_costs = [
            dose * constants.PRICE_PER_UI_FACTOR_VIII
            for dose in factor_consumption_list
        ]
    result_dict["total_factors_costs"] = np.sum(factor_costs) / (n_cycles / 52)
    result_dict["annual_factor_consumption"] = total_factor_use / (n_cycles / 52)
    result_dict["QALYS"] = total_utility_values
    return input_dict, result_dict


def on_demand_psa(n_samples: int, **kwargs):
    """
    Args:
        n_samples: Number of samples
        **kwargs: MarkovChain keyword arguments except transitions
    Returns:
        Tuple of (inputs, results) containing simulation inputs and outputs
    """
    # Define the problem for sampling
    problem = {"num_vars": 1, "names": ["ABR"], "bounds": [[0, 44]]}
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
        total=len(param_samples), desc="Simulating on_demand:", unit="simulation"
    )

    def update_bar(_):
        progress_bar.update(incr=1)

    with multiprocessing.Pool(processes=6) as pool:
        async_results = [
            pool.apply_async(
                on_demand_worker_function, args=(abr, kwargs), callback=update_bar
            )
            for abr in param_samples
        ]
        # Collect results
        for res in async_results:
            input_dict, result_dict = res.get()
            inputs.append(input_dict)
            results["total_factors_use"].append(result_dict["total_factors_use"])
            results["total_factors_costs"].append(result_dict["total_factors_costs"])
            results["annual_factor_consumption"].append(
                result_dict["annual_factor_consumption"]
            )
            results["QALYS"].append(result_dict["QALYS"])

    manager.stop()
    return inputs, results


def prophylaxis_worker_function(abr, kwargs: dict):
    """
    Worker function to process a single abr value.
    Args:
        abr: Input array for ABR value
        kwargs: MarkovChain keyword arguments
    Returns:
        Tuple of (input_dict, result_dict) for one iteration.
    """
    abr = round(float(abr[0]))  # array to float
    ajbr = round(0.75 * abr)
    builder = ProbabilityBuilder(abr=abr, ajbr=ajbr, annual_ltb_prob=0.0053)
    transition = builder.get_matrix()
    markov = MarkovChain(
        transitions=transition,
        lambda_bleeding=(abr - ajbr) / 52,  # PSA_KWARGS
        lambda_joint_bleeding=ajbr / 52,  # PSA_KWARGS
        **kwargs,
    )
    # Wrap utility_reward_function to track cumulative bleeds
    markov.add_reward_function(prophylaxis_factor_consumption)
    markov.add_reward_function(construct_utility_reward_function("prophylaxis"))
    markov.run()
    rewards = markov.collect_rewards()
    # Store results
    n_cycles = kwargs.get("steps")
    if not n_cycles or not isinstance(n_cycles, int):
        raise ValueError("Model number of steps not correctly defined for psa.")
    # Results
    input_dict = {"abr": abr, "ajbr": ajbr}
    result_dict = {}
    total_factor_use = np.sum(rewards[prophylaxis_factor_consumption.__name__][1:])
    total_utility_values = np.sum(rewards["utility_reward_function"][1:])
    result_dict["total_factors_use"] = total_factor_use
    factor_consumption_list: list = rewards[prophylaxis_factor_consumption.__name__][1:]
    # Discounted costs
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
        # Not discounted costs
        factor_costs = [
            dose * constants.PRICE_PER_UI_FACTOR_VIII
            for dose in factor_consumption_list
        ]
    result_dict["total_factors_costs"] = np.sum(factor_costs) / (n_cycles / 52)
    result_dict["annual_factor_consumption"] = total_factor_use / (n_cycles / 52)
    result_dict["QALYS"] = total_utility_values
    return input_dict, result_dict


def prophylaxis_psa(n_samples: int, **kwargs):
    """
    Args:
        n_samples: number of samples
        **kwargs: MarkovChain keyword arguments
    """
    # Mean: 3.66
    problem = {"num_vars": 1, "names": ["ABR"], "bounds": [[0, 28]]}
    param_samples = saltelli.sample(problem, n_samples, calc_second_order=False)
    inputs = []
    results = {
        "total_factors_use": [],
        "total_factors_costs": [],
        "annual_factor_consumption": [],
        "QALYS": [],
    }  # Placeholder
    manager = enlighten.get_manager()
    progress_bar: enlighten.Counter = manager.counter(
        total=len(param_samples), desc="Simulating prophylaxis:", unit="simulation"
    )

    def update_bar(_):
        progress_bar.update(incr=1)

    with multiprocessing.Pool(processes=6) as pool:
        async_results = [
            pool.apply_async(
                func=prophylaxis_worker_function,
                args=(abr, kwargs),
                callback=update_bar,
            )
            for abr in param_samples
        ]
        for res in async_results:
            input_dict, result_dict = res.get()
            inputs.append(input_dict)
            results["total_factors_use"].append(result_dict["total_factors_use"])
            results["total_factors_costs"].append(result_dict["total_factors_costs"])
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
    weight = cal_body_weight(step)
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
    weight = cal_body_weight(step)
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


def construct_utility_reward_function(treatment: Literal["on_demand", "prophylaxis"]):
    """Factory function to create a utility reward function with bleed tracking per treatment regimes."""
    cumulative_bleeds = [0]  # Use a list to allow modification in closure

    def utility_reward_function(step: int, state: str, **kwargs) -> float:
        """Returns utility values for each health state with event-based decay for healthy state."""
        # Increment cumulative bleeds based on the state at this step
        state_lower = state.lower()
        if state_lower in ["bleeding", "joint_bleeding"]:
            cumulative_bleeds[0] += 1
        decrement_per_bleed = (
            0.0018
            if treatment == "prophylaxis"
            else 0.0003725 if treatment == "on_demand" else None
        )
        if decrement_per_bleed is None:
            raise ValueError(f"Invalid treatment argument: {treatment}")
        utility_value = 0.0
        # Discounted utility
        if constants.DISCOUNT_RATE_WEEKLY:
            match state_lower:
                case "healthy":
                    # Base utility with decay based on cumulative bleeding events
                    base_utility = 0.85 if treatment == "on_demand" else 0.915
                    decrement = decrement_per_bleed * cumulative_bleeds[0]
                    adjusted_utility = max(
                        0.65, base_utility - decrement
                    )  # Minimum utility of 0.65
                    utility_value = (adjusted_utility * (1 / 52)) / (
                        1 + constants.DISCOUNT_RATE_WEEKLY
                    ) ** step
                case "bleeding":
                    utility_value = (0.60 * (1 / 52)) / (
                        1 + constants.DISCOUNT_RATE_WEEKLY
                    ) ** step
                case "joint_bleeding":
                    utility_value = (0.50 * (1 / 52)) / (
                        1 + constants.DISCOUNT_RATE_WEEKLY
                    ) ** step
                case "lt_bleeding":
                    utility_value = (0.25 * (1 / 52)) / (
                        1 + constants.DISCOUNT_RATE_WEEKLY
                    ) ** step
                case "death":
                    utility_value = 0
        # Not discounted utilities and not decremented
        else:
            match state_lower:
                case "healthy":
                    base_utility = 0.90
                    # Not decremented
                    # TODO:
                    # Consider adding arthropathy state some how
                    utility_value = base_utility * (1 / 52)
                case "bleeding":
                    utility_value = 0.60 * (1 / 52)
                case "joint_bleeding":
                    utility_value = 0.50 * (1 / 52)
                case "lt_bleeding":
                    utility_value = 0.25 * (1 / 52)
                case "death":
                    utility_value = 0
        return utility_value

    return utility_reward_function
