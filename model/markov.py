from typing import List, Union, Generator, Optional, Callable, Dict
from model.utils import cal_body_weight
from model.utils import probability_at_least_one_event
from model import constants
from pathlib import Path
from SALib.sample import saltelli
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MarkovChain:
    """A Markov chain implementation that generates state transitions with reward tracking."""

    def __init__(
        self,
        states: List[str],
        transitions: Union[List[List[np.float64]], np.ndarray],
        start_state: str,
        steps: int,
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

        # Calculate reward for initial state (step 0)
        if self.reward_functions:
            for func in self.reward_functions:
                r = func(step=0, state=self.states[current_state])
                self.rewards[func.__name__].append(r)

        for step in range(self.steps):
            # Yield current state
            yield self.states[current_state]

            # Transition to next state
            probs = self.transitions[current_state]
            current_state = np.random.choice(self.num_states, p=probs)

            # Calculate rewards for the new state
            if self.reward_functions:
                for func in self.reward_functions:
                    r = func(step=step, state=self.states[current_state])
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

    def to_minor(self):
        return probability_at_least_one_event(self.aebr, "annual")

    def to_major(self):
        return probability_at_least_one_event(self.ajbr, "annual")

    def to_ltb(self):
        return probability_at_least_one_event(self.weekly_ltb_prob, "weekly")

    def to_healthy(self):
        total = self.to_ltb() + self.to_minor() + self.to_major()
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
                self.to_minor(),
                self.to_major(),
                self.to_ltb(),
                0,
            ],  # Row: Healthy
            [
                self.to_healthy(),
                self.to_minor(),
                self.to_major(),
                self.to_ltb(),
                0,
            ],  # Row: Minor
            [
                self.to_healthy(),
                self.to_minor(),
                self.to_major(),
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


def on_demand_psa(n_samples: int, **kwargs):
    """
    Args:
        n_samples: number of samples
        **kwargs: MarkovChain keyword arguments except transitions
    """
    # Starts from 12 (?) or 0
    problem = {"num_vars": 1, "names": ["ABR"], "bounds": [[0, 44]]}
    param_samples = saltelli.sample(problem, n_samples, calc_second_order=False)

    inputs = []
    results = {
        "total_factors_use": [],
        "total_factors_costs": [],
        "annual_factor_consumption": [],
        "QALYS": [],
    }  # Placeholder
    for abr in param_samples:
        abr = round(float(abr[0]))  # array to float
        ajbr = round(0.75 * abr)
        builder = ProbabilityBuilder(abr=abr, ajbr=ajbr, annual_ltb_prob=0.021)
        transition = builder.get_matrix()
        markov = MarkovChain(transitions=transition, **kwargs)
        markov.add_reward_function(on_demand_factor_consumption)
        markov.run()
        rewards = markov.collect_rewards()

        # Store results
        n_cycles = kwargs.get("steps")
        inputs.append({"abr": abr, "ajbr": ajbr})
        total_factor_use = np.sum(rewards[on_demand_factor_consumption.__name__])
        results["total_factors_use"].append(total_factor_use)
        results["total_factors_costs"].append(
            total_factor_use
            * constants.PRICE_PER_UI_FACTOR_VIII
            / constants.PPP_CONVERSION_FACTOR
        )
        results["annual_factor_consumption"].append(total_factor_use / n_cycles * 12)
    return inputs, results


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
    for abr in param_samples:
        abr = round(float(abr[0]))  # array to float
        ajbr = round(0.75 * abr)
        builder = ProbabilityBuilder(abr=abr, ajbr=ajbr, annual_ltb_prob=0.0053)
        transition = builder.get_matrix()
        markov = MarkovChain(transitions=transition, **kwargs)
        markov.add_reward_function(prophylaxis_factor_consumption)
        markov.run()
        rewards = markov.collect_rewards()

        # Store results
        n_cycles = kwargs.get("steps")
        inputs.append({"abr": abr, "ajbr": ajbr})
        total_factor_use = np.sum(rewards[prophylaxis_factor_consumption.__name__])
        results["total_factors_use"].append(total_factor_use)
        results["total_factors_costs"].append(
            total_factor_use
            * constants.PRICE_PER_UI_FACTOR_VIII
            / constants.PPP_CONVERSION_FACTOR
        )
        results["annual_factor_consumption"].append(total_factor_use / n_cycles * 12)
    return inputs, results


# TODO:
# Make sure dosing are tuned


def on_demand_factor_consumption(step: int, state: str):
    """Example reward function that calculates and prints body weight."""
    weight = cal_body_weight(step)
    injected_dose = 0
    if state.lower() == "minor":
        # Muscle | illopsoas | Renal | Oral mucosa and dental
        injected_dose = round(weight * 90)  # (30 * 3)
    elif state.lower() == "major":
        # Joint
        injected_dose = round(weight * 60)  # (30 * 2)
    elif state.lower() == "lt_bleeding":
        # Intra_cranial | Gastro | Neck & throat
        injected_dose = round(weight * 500)
    return injected_dose


def prophylaxis_factor_consumption(step: int, state: str):
    """Example reward function that calculates and prints body weight."""
    weight = cal_body_weight(step)
    # TODO:
    # Modified prophylaxis? PSA
    injected_dose = round(weight * 25 * 2)  # (Standard prophylaxis dosing)
    if state.lower() == "minor":
        # Muscle | illopsoas | Renal | Oral mucosa and dental
        injected_dose += round(weight * 90)  # (30 * 3)
    elif state.lower() == "major":
        # Joint
        injected_dose += round(weight * 60)  # (30 * 2)
    elif state.lower() == "lt_bleeding":
        # Intra_cranial | Gastro | Neck & throat
        injected_dose += round(weight * 500)
    return injected_dose
