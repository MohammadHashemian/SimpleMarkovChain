from pydantic import BaseModel
from typing import Literal, Callable, List, Dict
from enum import StrEnum
from model import constants
from model.markov_chain import TransitionGenerator, MarkovChains, Chain
from model.visualization import visualize_abr
import model.utils as model_util
from SALib.sample import saltelli
import multiprocessing
import enlighten
import numpy as np


class Treatment(StrEnum):
    ON_DEMAND = "on_demand"
    PROPHYLAXIS = "prophylaxis"


class MarkovResult(BaseModel):
    initial_state: str
    final_state: str
    steps: int
    path: List[str]


class HemophiliaResult(MarkovResult):
    total_factor_use: float
    total_factor_costs: float
    annual_factor_consumption: float
    annual_factor_costs: float
    hemarthrosis: float
    qaly: float
    abr: float


def generate_population_abrs(
    strategy: Literal["on_demand", "prophylaxis"],
    n_samples: int = 64,
    visualize: bool = True,
):
    """
    Samples ABR values to figure the real world patient abrs distribution

    Args:
        strategy: Either "on_demand" or "prophylaxis".
        n_samples: Number of samples.

    Returns:
        None
    """
    if strategy not in ["on_demand", "prophylaxis"]:
        raise ValueError("Invalid strategy")
    # Published articles data for annual bleeding rate reported as (Mean, SD)
    study_data = np.array(
        constants.ON_DEMAND_ABR_REPORTS
        if strategy == "on_demand"
        else constants.PROPHYLAXIS_ABR_REPORTS
    )
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

    abr_values = np.array([weights_to_abr(params) for params in param_samples])
    visualize_abr(abr_values, strategy) if visualize else None
    return abr_values


def worker_function(
    markov: MarkovChains,
    worker_kwargs: Dict,
) -> tuple[Dict, HemophiliaResult]:
    """
    Worker function for simulating Markov chains for a given ABR and strategy.

    Args:
        worker_kwargs: Worker function keyword arguments.
        markov_chain: MarkovChains instance to run

    Returns:
        Tuple of (input_dict, Results) containing simulation inputs and outputs.
    """
    # Unwrap kwargs
    treatment: Treatment | None = worker_kwargs.get("treatment", None)
    if treatment is None or treatment not in Treatment:
        raise KeyError("Required argument not passed to worker function")
    abr: np.float64 | None = worker_kwargs.get("abr", None)
    if abr is None or not isinstance(abr, np.float64):
        raise KeyError("Required arguments not passed to worker function")

    # Annual values
    ajbr = abr * constants.AJBR_FRACTION
    altb = abr * constants.LTB_FRACTION
    aebr = abr - (ajbr + altb)
    amr = constants.MORTALITY_RATE

    # Weekly values
    wbr = abr / constants.WOY  # weekly bleeding rate
    wjbr = ajbr / constants.WOY  # weekly joint bleeding rate
    wltb = altb / constants.WOY  # weekly life-threatening rate
    webr = aebr / constants.WOY  # weekly non-joint bleeding rate
    wmr = amr / constants.WOY  # weekly frequency of natural dying

    # Direct probability assignment for no_bleeding event
    weekly_no_event_prob = np.exp(-wbr)

    chains_map = markov.chains_map
    main_chain: Chain | None = chains_map.get("main", None)
    if not main_chain:
        raise ValueError("Main chain is not provided to start simulation")

    # ---- Transition pairs ----
    # Self transitions should be avoided, probability of staying will be calculated as survival probability
    transition_pairs = {
        # Healthy Transitions (competing risks)
        ("Healthy", "Bleeding"): (webr, "weekly"),
        ("Healthy", "Hemarthrosis"): (wjbr, "weekly"),
        ("Healthy", "LT_Bleeding"): (wltb, "weekly"),
        ("Healthy", "Death"): (wmr, "weekly"),
        # Bleeding Transitions (competing risks)
        ("Bleeding", "Healthy"): (weekly_no_event_prob, None),  # Direct probability
        ("Bleeding", "Hemarthrosis"): (wjbr, "weekly"),
        ("Bleeding", "LT_Bleeding"): (wltb, "weekly"),
        ("Bleeding", "Death"): (wmr, "weekly"),
        # Hemarthrosis Transitions (competing risks)
        ("Hemarthrosis", "Healthy"): (weekly_no_event_prob, None),  # Direct probability
        ("Hemarthrosis", "Bleeding"): (webr, "weekly"),
        ("Hemarthrosis", "LT_Bleeding"): (wltb, "weekly"),
        ("Hemarthrosis", "Death"): (wmr, "weekly"),
    }

    # States: ["Healthy", "Bleeding", "Hemarthrosis", "LT_Bleeding", "Death"]
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
    main_chain.matrix = np.array(transition_builder.get_crm())

    if len(markov.worker_kwargs.keys()) != 0:
        raise ValueError("Markov chain already assigned with PSA keyword arguments")
    markov.worker_kwargs = {
        "treatment": treatment,
        "wbr": wbr,
        "wjbr": wjbr,
        "webr": webr,
    }

    markov.add_store_function("number_of_bleeds", model_util.count_bleeds)
    markov.add_store_function("number_of_hemarthrosis", model_util.count_hemarthrosis)
    markov.add_reward_function(factor_consumption)
    markov.add_reward_function(construct_utility_reward_function(treatment))
    sequences = markov.run()
    rewards = markov.collect_rewards()

    # ---- Results ----
    n_cycles = markov.steps

    input_dict = {
        "abr": abr,
        "ajbr": wjbr * constants.WOY,
        "chains": markov.chains,
    }

    _utility_func_name = construct_utility_reward_function.__name__.removeprefix(
        "construct_"
    )
    sum_factor_consumption = np.sum(rewards[factor_consumption.__name__][1:])
    sum_utilities = np.sum(rewards[_utility_func_name][1:])
    factor_sequence: list = rewards[factor_consumption.__name__][1:]

    def _to_cost(
        dose: float | int,
        n: int,
        unit: Literal["IRR", "USD"] = constants.REPORT_UNIT,
        ppp: bool = constants.REPORT_PPP,
    ):
        if isinstance(dose, int):
            dose = float(dose)

        if unit not in ["USD", "IRR"]:
            raise ValueError(f"Report unit meant to be USD or IRR, got {unit}")

        cost = dose * constants.PRICE_PER_UI_FACTOR_VIII
        if unit == "IRR":
            cost = cost
        elif unit == "USD" and not ppp:
            cost = cost / constants.RIAL_USD_PRICE
        elif unit == "USD" and ppp:
            cost = cost / constants.PPP_CONVERSION_FACTOR

        if constants.DISCOUNT_RATE_WEEKLY:
            cost = cost / (1 + constants.DISCOUNT_RATE_WEEKLY) ** n
        return cost

    def to_annual(array: List | np.ndarray):
        if isinstance(array, List):
            array = np.array(array)
        return array / (n_cycles / constants.WOY)

    # Costs (corrected for consistent discounting and currency conversion)
    factor_costs = [_to_cost(dose, i) for i, dose in enumerate(factor_sequence)]
    # Store aggregated results
    output = HemophiliaResult(
        initial_state=markov.entrance,
        final_state=sequences[-1:][0],
        steps=n_cycles,
        path=sequences,
        total_factor_use=sum_factor_consumption,
        total_factor_costs=np.sum(factor_costs),
        annual_factor_consumption=to_annual(sum_factor_consumption), # type: ignore
        annual_factor_costs=np.sum(to_annual(factor_costs)),
        qaly=sum_utilities,
        abr=np.sum(to_annual(rewards["number_of_bleeds"])),
        hemarthrosis=np.sum(to_annual(rewards["number_of_hemarthrosis"])),
    )
    return (input_dict, output)


def psa_wrapper(
    simulation_name: str,
    worker_inputs: list,
    worker_func: Callable,
    markov_chain: MarkovChains,
) -> tuple[list, list]:
    """
    Args:
        markov_chain: markov chain class instance to parallelize
    """
    model_inputs = []
    model_outputs = []

    manager = enlighten.get_manager()
    progress_bar: enlighten.Counter = manager.counter(
        total=len(worker_inputs),
        desc=f"Simulating {simulation_name}:",
        unit="simulation",
    )

    def update_bar(_):
        progress_bar.update(incr=1)

    def error_handler(e: BaseException):
        raise ValueError(f"simulation failed {e}")

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        async_results = [
            pool.apply_async(
                func=worker_func,
                args=(markov_chain, worker_kwargs),
                callback=update_bar,
                error_callback=error_handler,
            )
            for worker_kwargs in worker_inputs
        ]
        for res in async_results:
            input_dict, output = res.get()
            model_inputs.append(input_dict)
            model_outputs.append(output)
    manager.stop()
    return model_inputs, model_outputs


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
    ssp = constants.SHORT_SIMULATION_START_AGE_IN_WEEK
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


def construct_utility_reward_function(
    treatment: Literal["on_demand", "prophylaxis"],
) -> Callable[[int, str], float]:
    """
    Factory function to create a utility reward function.
    Tracks bleeds and applies discounting or utility decrement
    """
    # Closure mutable containers
    bleeds_count = [0]
    hemarthrosis_count = [0]

    def get_pettersson(value: float) -> int:
        """Converts hemarthrosis bleeding count to pettersson joint health score"""
        score = round(value / constants.PETTERSSON_CONVERSION_FACTOR)
        if score >= 79:
            return 79
        return int(score)

    def discount(value: float, step: int) -> float:
        """Apply weekly discounting if enabled."""
        if constants.DISCOUNT_RATE_WEEKLY:
            return value / (1 + constants.DISCOUNT_RATE_WEEKLY) ** step
        return value

    def utility_reward_function(
        step: int,
        state: str,
        **kwargs,
    ) -> float:
        """Returns utility values for each health state with event-based decay for healthy state."""
        nonlocal bleeds_count
        nonlocal hemarthrosis_count
        state_lower = state.lower()

        # Increment cumulative bleeds based on the state at this step
        try:
            bleeds = kwargs["number_of_bleeds"]
            hemarthrosis = kwargs["number_of_hemarthrosis"]
            if state_lower in ["bleeding", "hemarthrosis"]:
                bleeds_count[0] += bleeds
            if state_lower == "hemarthrosis":
                hemarthrosis_count[0] += hemarthrosis
        except KeyError:
            raise ValueError(
                "Keyword arguments are not present within utility reward function"
            )

        if state_lower == "healthy":
            score = get_pettersson(hemarthrosis_count[0])
            category = constants.PETTERSSON_CATEGORIES[score]
            utility = constants.STATE_UTILITIES[category]
        else:
            utility = constants.STATE_UTILITIES[state]

        # Normalize to per-week utility and apply discount
        return discount(utility * (1 / constants.WOY), step)

    return utility_reward_function
