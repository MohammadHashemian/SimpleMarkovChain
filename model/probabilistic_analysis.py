from typing import List, Unpack
from model import constants
from model.defined_types import Currencies
from model.markov_chain import MarkovChains, Chain, TransitionGenerator
from model.defined_types import (
    HemophiliaInput,
    HemophiliaOutput,
    HemophiliaRewardArgs,
    Treatment,
)
from model.utils import zero_truncated_mass_function_numba
from model.visualization import visualize_abr
import model.utils as model_util
from SALib.sample import saltelli
import numpy as np


def sample_population_abrs(
    study_data: list["tuple"],
    n_samples: int = 64,
    visualize: bool = True,
) -> list[float]:
    """
    Samples ABR values to figure the real world patient abrs distribution

    Args:
        - study_data: published articles data for annual bleeding rate reported as (Mean, SD)
        - n_samples: base sample size for sobol sampler

    Returns:
        list[float]: array of sampled annual bleeding rates
    """
    #
    num_studies = len(study_data)
    # Dynamic lengthening
    problem = {
        "num_vars": num_studies,
        "names": [f"study_{i}_weight" for i in range(num_studies)],
        "bounds": [[0, 1]] * num_studies,
    }
    param_samples = saltelli.sample(problem, n_samples, calc_second_order=True)

    def to_abr(weights) -> float:
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

    abr_values = [to_abr(params) for params in param_samples]
    visualize_abr(abr_values) if visualize else None
    return abr_values


def worker_function(
    markov: MarkovChains,
    worker_kwargs: HemophiliaInput,
) -> tuple[HemophiliaInput, HemophiliaOutput]:
    """
    Single markov model initializer function for given annual bleeding rate and treatment arm

    Args:
        worker_kwargs: Worker function keyword arguments
        markov_chain: MarkovChains instance to run

    Returns:
        Tuple of (HemophiliaInput, HemophiliaOutput) containing simulation inputs and outputs
    """
    # ---- Unwrap key word arguments ----
    treatment: Treatment | None = worker_kwargs.get("treatment", None)
    if treatment is None or treatment not in Treatment:
        raise KeyError("Required argument not passed to worker function")
    abr: np.float64 | float = worker_kwargs.get("abr")
    if abr is None or not isinstance(abr, np.float64 | float):
        raise KeyError("Required arguments not passed to worker function")

    chains_map = markov.chains_map
    main_chain: Chain | None = chains_map.get("main", None)
    if not main_chain:
        raise ValueError("Main chain is not provided to start simulation")

    # ---- Constructing Transition Pairs ----
    def _to_weekly(annual_rate: float) -> float:
        return annual_rate / constants.WOY

    # Annual values
    annual_rates = {
        "abr": abr,
        "ajbr": abr * constants.AJBR_FRACTION,
        "altb": abr * constants.LTB_FRACTION,
        "aebr": (
            abr - ((abr * constants.AJBR_FRACTION) + (abr * constants.LTB_FRACTION))
        ),
        "amr": constants.CRUDE_MORTALITY_RATE,
    }

    # Direct probability assignment for no_bleeding event
    weekly_no_event_prob = np.exp(-_to_weekly(abr))

    # Self transitions should be avoided, probability of staying will be calculated as survival probability inside TransitionGenerator build function
    transition_pairs = {
        # Healthy Transitions (competing risks)
        ("Healthy", "Bleeding"): (_to_weekly(annual_rates["aebr"]), "weekly"),
        ("Healthy", "Hemarthrosis"): (_to_weekly(annual_rates["ajbr"]), "weekly"),
        ("Healthy", "LT_Bleeding"): (_to_weekly(annual_rates["altb"]), "weekly"),
        ("Healthy", "Death"): (_to_weekly(annual_rates["amr"]), "weekly"),
        # Bleeding Transitions (competing risks)
        ("Bleeding", "Healthy"): (weekly_no_event_prob, None),  # Direct probability
        ("Bleeding", "Hemarthrosis"): (_to_weekly(annual_rates["ajbr"]), "weekly"),
        ("Bleeding", "LT_Bleeding"): (
            _to_weekly(annual_rates["altb"]) + _to_weekly(annual_rates["amr"]),
            "weekly",
        ),
        ("Bleeding", "Death"): (_to_weekly(annual_rates["amr"]), "weekly"),
        # Hemarthrosis Transitions (competing risks)
        ("Hemarthrosis", "Healthy"): (weekly_no_event_prob, None),  # Direct probability
        ("Hemarthrosis", "Bleeding"): (_to_weekly(annual_rates["aebr"]), "weekly"),
        ("Hemarthrosis", "LT_Bleeding"): (_to_weekly(annual_rates["altb"]), "weekly"),
        ("Hemarthrosis", "Death"): (_to_weekly(annual_rates["amr"]), "weekly"),
    }

    # States order: ["Healthy", "Bleeding", "Hemarthrosis", "LT_Bleeding", "Death"]
    special_transitions = {
        "Death": [0.0] * (len(main_chain.states) - 1) + [1.0],  # Absorbing
        "LT_Bleeding": [0.94] + [0.0] * (len(main_chain.states) - 2) + [0.06],
    }

    # ---- Build transition matrices ----
    transition_builder = TransitionGenerator(
        states=main_chain.states,
        transition_pairs=transition_pairs,
        special_transitions=special_transitions,
    )
    main_chain.matrix = np.array(transition_builder.build())

    if len(markov.worker_kwargs.keys()) != 0:
        raise ValueError("Markov chain already assigned with PSA keyword arguments")

    input_dict: HemophiliaInput = {
        "treatment": treatment,
        "abr": abr,
        "ajbr": annual_rates["ajbr"],
        "wbr": _to_weekly(annual_rates["abr"]),
        "wjbr": _to_weekly(annual_rates["ajbr"]),
        "webr": _to_weekly(annual_rates["aebr"]),
    }
    markov.worker_kwargs = {**input_dict}

    markov.add_store_function("number_of_bleeds", count_bleeds)
    markov.add_store_function("number_of_hemarthrosis", count_bleeds)
    markov.add_store_function("pettersson_score", construct_pettersson_calculator())
    markov.add_reward_function(factor_consumption)
    markov.add_reward_function(utility_reward_function)
    sequences = markov.run()
    rewards = markov.collect_rewards()

    # ---- Results ----
    n_cycles = markov.steps

    sum_factor_consumptions = np.sum(rewards[factor_consumption.__name__][1:])
    sum_utilities = np.sum(rewards[utility_reward_function.__name__][1:])
    factor_sequence: list = rewards[factor_consumption.__name__][1:]

    def _to_cost(
        dose: float | int,
        n: int,
        unit: Currencies = constants.MODEL_CURRENCY,
        ppp: bool = constants.REPORT_PPP,
    ):
        if isinstance(dose, int):
            dose = float(dose)

        if unit not in Currencies:
            raise ValueError(
                "Can not calculate correct costs. wrong currency supplied."
            )

        cost = dose * constants.PRICE_PER_UI_FACTOR_VIII
        if unit.upper() == "IRR":
            cost = cost
        elif unit.upper() == "TOMAN":
            cost = cost / 10
        elif unit == "USD" and not ppp:
            cost = cost / constants.RIAL_USD_PRICE
        elif unit == "USD" and ppp:
            cost = cost / constants.PPP_CONVERSION_FACTOR

        if constants.DISCOUNT_RATE_WEEKLY:
            cost = cost / (1 + constants.DISCOUNT_RATE_WEEKLY) ** n
        return cost

    def _to_annual(array: List | np.ndarray):
        if isinstance(array, List):
            array = np.array(array)
        return array / (n_cycles / constants.WOY)

    # Costs (corrected for consistent discounting and currency conversion)
    factor_costs = [_to_cost(dose, i) for i, dose in enumerate(factor_sequence)]
    # Store simulation results
    output_dict = HemophiliaOutput(
        initial_state=markov.entrance,
        final_state=sequences[-1:][0],
        steps=n_cycles,
        path=sequences,
        factor_consumption=sum_factor_consumptions,
        factor_costs=np.sum(factor_costs),
        annual_factor_consumption=_to_annual(sum_factor_consumptions),  # type: ignore
        annual_factor_costs=np.sum(_to_annual(factor_costs)),
        hemarthrosis=_to_annual(np.sum(rewards["number_of_hemarthrosis"])),  # type: ignore
        qaly=sum_utilities,
        abr=_to_annual(np.sum(rewards["number_of_bleeds"])),  # type: ignore
        pettersson_score=rewards["pettersson_score"],
    )
    return (input_dict, output_dict)


def factor_consumption(
    step: int,
    state: str,
    **kwargs,
) -> float:
    """
    Reward function that calculates factor VIII consumption per patient body weight over time.

    Args:
        step (int): Current markov cycle step.
        state (str): Current active state to be rewarded
        kwargs: Contains argument passed to markov instance from worker function to markov chains to reward function

    Returns:
        int: Total factor VIII consumption (IU).
    """
    # Unwrapping arguments
    try:
        treatment: Treatment | None = kwargs["treatment"]
        number_of_bleeds: int | None = kwargs["number_of_bleeds"]
    except KeyError:
        raise KeyError(
            f"Keyword arguments are not bounded to factor consumption reward function -> {kwargs}"
        )
    if treatment not in Treatment:
        raise ValueError(
            f"Argument treatment passed to factor reward function expected to be in Literal, got {type(treatment)}"
        )
    if not isinstance(number_of_bleeds, int):
        raise ValueError(
            f"Argument number_of_bleeds passed to factor reward function expected to be in Type int, got {type(number_of_bleeds)}"
        )

    # Starting treatment from age 2 yo
    ssp = constants.PEDIATRIC_STARTING_POINT
    weight = model_util.cal_body_weight(step, b=ssp)
    injected_dose = 0

    # Add prophylaxis baseline if applicable
    injected_dose += (
        round(weight * constants.STANDARD_PROPHYLAXIS_WEEKLY_DOSE, 2)
        if treatment == "prophylaxis"
        else 0
    )

    state_lower = state.lower()
    # Bleeding-related consumption
    match state_lower:
        case "bleeding":
            bd = constants.BLEEDING_DOSE
            injected_dose += round(weight * bd) * number_of_bleeds
        case "hemarthrosis":
            jbd = constants.JOINT_BLEEDING_DOSE
            injected_dose += round(weight * jbd) * number_of_bleeds
        case "lt_bleeding":
            ltbd = constants.LT_BLEEDING_DOSE
            injected_dose += round(weight * ltbd)
        case "death":
            # to avoid assigning base prophylaxis dose to dead states
            injected_dose = 0
    return injected_dose


def utility_reward_function(
    step: int, state: str, **kwargs: Unpack[HemophiliaRewardArgs]
) -> float:
    """Returns utility values for each health state with event-based decay for healthy state."""

    def discount(value: float, step: int) -> float:
        """Apply weekly discounting if enabled."""
        if constants.DISCOUNT_RATE_WEEKLY is not None:
            return value / (1 + constants.DISCOUNT_RATE_WEEKLY) ** step
        return value

    state_lower = state.lower()

    if state_lower == "healthy":
        score = kwargs.get("pettersson_score", None)
        if score is None:
            raise ValueError(
                "No pettersson_score argument passed to utility reward function"
            )
        category = constants.PETTERSSON_CATEGORIES[score]
        utility = constants.STATE_UTILITIES[category]
        # Additional disutility when bleeding occurs in severe arthropathy
        # as severe_arthropathy utility is less than bleeding by default
        if category.lower() == "severe_arthropathy" and state_lower == "bleeding":
            utility -= constants.SEVERE_ARTHROPATHY_BLEEDING_DISUTILITY
    else:
        utility = constants.STATE_UTILITIES[state]

    # Normalize to per-week utility and apply discount
    return discount(utility * (1 / constants.WOY), step)


def count_bleeds(
    state: str, k_range: int = 8, **kwargs: Unpack[HemophiliaRewardArgs]
) -> int:
    """
    Calculate the number of bleeding events in a given state using zero truncated poisson distribution.

    Args:
        state (str): Current patient state (e.g., 'Bleeding', 'Hemarthrosis').
        **kwargs: Additional parameters, including 'wbr','webr' and 'wjbr' to calculate lambda.

    Returns:
        int: Number of bleeding events on a single week (interval).
    """
    state_lower = state.lower()
    if state_lower != "bleeding" and state_lower != "hemarthrosis":
        return 0

    def calculate_weights():
        lam = kwargs.get("wbr", None)
        if not isinstance(lam, (int, float, np.float64)) or lam < 0:
            raise ValueError(f"Invalid λ for state {state}: {lam}, type:{type(lam)}")
        k_array = np.arange(start=1, stop=k_range, step=1).astype(int)
        weights = np.array(
            [zero_truncated_mass_function_numba(lam=lam, k=k) for k in k_array]
        )
        sum_of_weights = weights.sum()
        if sum_of_weights <= 0:
            raise ValueError("Probabilities sum to non-positive value")
        normalized_weights = weights / sum_of_weights
        samples = np.random.choice(k_array, p=normalized_weights, size=1).astype(int)
        if samples.shape[0] > 1:
            raise ValueError("Sample size other than 1, is not acceptable")
        sample = samples[0]
        return int(sample)

    return calculate_weights()


def count_hemarthrosis(
    state: str, k_range=8, **kwargs: Unpack[HemophiliaRewardArgs]
) -> int:
    if state.lower() != "hemarthrosis":
        return 0
    return count_bleeds(state, k_range, **kwargs)


def construct_pettersson_calculator():
    """
    Factory function to create a pettersson reward function
    Tracks hemarthrosis
    """
    # Closure mutable containers
    hemarthrosis_count = [0]

    def to_pettersson_score(total_hemarthrosis: float) -> int:
        """Converts hemarthrosis bleeding count to pettersson joint health score"""
        score = round(total_hemarthrosis / constants.PETTERSSON_CONVERSION_FACTOR, 0)
        if score >= 79:
            return 79
        return int(score)

    def pettersson_score(state: str, **kwargs: Unpack[HemophiliaRewardArgs]) -> float:
        nonlocal hemarthrosis_count
        hemarthrosis = kwargs.get("number_of_hemarthrosis", None)
        if hemarthrosis is None:
            raise ValueError(
                "Argument number_of_hemarthrosis not bounded to pettersson score reward function"
            )
        state_lower = state.lower()
        # Increment bleeds base on state
        if state_lower == "hemarthrosis":
            hemarthrosis_count[0] += hemarthrosis
        return to_pettersson_score(hemarthrosis_count[0])

    return pettersson_score
