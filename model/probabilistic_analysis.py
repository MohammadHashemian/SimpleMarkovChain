from typing import Literal, Callable
from model import constants
from model.markov_chain import TransitionGenerator, MarkovChains, Results
from model.visualization import visualize_abr
from model.utils import cal_body_weight, count_bleeds_poisson
from SALib.sample import saltelli
import numpy as np
import enlighten
import multiprocessing


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
        Tuple of (input_dict, Results) containing simulation inputs and outputs.
    """
    # Annual values
    annual_abr = abr
    annual_ajbr = annual_abr * constants.AJBR_FRACTION
    annual_ltb = annual_abr * constants.LTB_FRACTION
    annual_aebr = annual_abr - (annual_ajbr + annual_ltb)
    annual_mortality = constants.MORTALITY_RATE

    # Weekly values
    wbr = annual_abr / constants.WOY  # weekly bleeding rate
    wjbr = annual_ajbr / constants.WOY  # weekly joint bleeding rate
    wltb = annual_ltb / constants.WOY  # weekly life-threatening rate
    webr = annual_aebr / constants.WOY  # weekly non-joint bleeding rate
    wmr = annual_mortality / constants.WOY  # weekly frequency of natural dying

    # Direct probability assignment for no_bleeding event
    weekly_no_event_prob = np.exp(-wbr)

    chains = kwargs["chains"]
    primary_states = chains["primary"][0]
    secondary_states = chains["secondary"][0]

    # ---- Transition pairs (shared across both) ----
    primary_transition_pairs = {
        # Healthy Transitions (competing risks)
        ("Healthy", "Bleeding"): (webr, "weekly"),
        ("Healthy", "Hemarthrosis"): (wjbr, "weekly"),
        ("Healthy", "LT_Bleeding"): (wltb, "weekly"),
        ("Healthy", "Death"): (wmr, "weekly"),
        # Bleeding Transitions (competing risks)
        ("Bleeding", "Healthy"): (weekly_no_event_prob, None),  # Direct probability
        ("Bleeding", "Bleeding"): (webr, "weekly"),
        ("Bleeding", "Hemarthrosis"): (wjbr, "weekly"),
        ("Bleeding", "LT_Bleeding"): (wltb, "weekly"),
        ("Bleeding", "Death"): (wmr, "weekly"),
        # Hemarthrosis Transitions (competing risks)
        ("Hemarthrosis", "Healthy"): (weekly_no_event_prob, None),  # Direct probability
        ("Hemarthrosis", "Bleeding"): (webr, "weekly"),
        ("Hemarthrosis", "LT_Bleeding"): (wltb, "weekly"),
        ("Hemarthrosis", "Arthropathy"): (constants.EARLY_ARTHROPATHY, "weekly"),
        ("Hemarthrosis", "Death"): (wmr, "weekly"),
        # SELF_TRANSITIONS
        # ("Healthy", "Healthy"): (no_event_weekly, "weekly"),
        # ("Bleeding", "Healthy"): (no_event_weekly, "weekly"),
        # ("Hemarthrosis", "Hemarthrosis"): (wjbr, "weekly"),
    }

    secondary_transition_pairs = {
        # Arthropathy Transitions (competing risks)
        ("Arthropathy", "Bleeding"): (webr, "weekly"),
        ("Arthropathy", "Hemarthrosis"): (wjbr, "weekly"),
        ("Arthropathy", "LT_Bleeding"): (wltb, "weekly"),
        ("Arthropathy", "Death"): (wmr, "weekly"),
        # Bleeding Transitions (competing risks)
        ("Bleeding", "Arthropathy"): (weekly_no_event_prob, None),  # Direct probability
        ("Bleeding", "Hemarthrosis"): (wjbr, "weekly"),
        ("Bleeding", "LT_Bleeding"): (wltb, "weekly"),
        ("Bleeding", "Death"): (wmr, "weekly"),
        # Hemarthrosis Transitions (competing risks)
        ("Hemarthrosis", "Arthropathy"): (
            weekly_no_event_prob,
            None,
        ),  # Direct probability
        ("Hemarthrosis", "Bleeding"): (webr, "weekly"),
        ("Hemarthrosis", "LT_Bleeding"): (wltb, "weekly"),
        ("Hemarthrosis", "Death"): (wmr, "weekly"),
        # SELF_TRANSITIONS
        # ("Arthropathy", "Arthropathy"): (no_event_weekly, "weekly"),
        # ("Bleeding", "Bleeding"): (aebr_weekly, "weekly"),
        # ("Hemarthrosis", "Hemarthrosis"): (wjbr, "weekly"),
    }

    # Primary states: ["Healthy", "Bleeding", "Hemarthrosis", "Arthropathy", "LT_Bleeding", "Death"]
    primary_special_transitions = {
        "Death": [0.0] * (len(primary_states) - 1) + [1.0],  # Absorbing
        "LT_Bleeding": [0.94] + [0.0] * (len(primary_states) - 2) + [0.06],
    }
    secondary_special_transitions = {
        "Death": [0.0] * (len(secondary_states) - 1) + [1.0],  # Absorbing
        "LT_Bleeding": [0.94] + [0.0] * (len(secondary_states) - 2) + [0.06],
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
    chains["primary"] = (primary_states, primary_builder.get_crm())
    chains["secondary"] = (secondary_states, secondary_builder.get_crm())

    # ---- Markov model setup ----
    markov = MarkovChains(
        chains=chains,
        wbr=wbr,  # Weekly bleeding rate
        wjbr=wjbr,  # Corrected: weekly hemarthrosis rate
        webr=webr,  # Corrected: weekly non-joint bleeding rate
        **{k: v for k, v in kwargs.items() if k != "chains"},
    )

    # Select reward functions based on treatment
    match strategy:
        case "on_demand":
            factor_func = on_demand_factor_consumption
        case "prophylaxis":
            factor_func = prophylaxis_factor_consumption
        case _:
            raise ValueError(f"Invalid treatment: {strategy}")

    markov.add_store_function(arg_name="number_of_bleeds", func=count_bleeds_poisson)
    markov.add_reward_function(factor_func)
    markov.add_reward_function(construct_utility_reward_function(strategy))
    sequences = markov.run()
    rewards = markov.collect_rewards()

    # ---- Results ----
    n_cycles = kwargs.get("steps")
    if not n_cycles or not isinstance(n_cycles, int):
        raise ValueError("Model number of steps not correctly defined for psa.")

    inputs = {"abr": abr, "ajbr": wjbr * constants.WOY, "chains": chains}

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
    results = Results(
        total_factor_use=total_factor_use,
        total_factor_costs=np.sum(factor_costs),
        annual_factor_consumption=total_factor_use / (n_cycles / constants.WOY),
        annual_factor_costs=np.sum(factor_costs) / (n_cycles / constants.WOY),
        qaly=total_utility_values,
        sequences=sequences,
        abr=np.sum(rewards["number_of_bleeds"]) / (n_cycles / constants.WOY),
    )
    return (inputs, results)


def markov_chains_psa_wrapper(
    strategy: str, n_samples: int, **kwargs
) -> tuple[list, list[Results]]:
    """
    Run probabilistic sensitivity analysis simulation for a given treatment strategy.

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
    weight = cal_body_weight(step, b=constants.SHORT_SIMULATION_START_AGE_IN_WEEK)
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
        case "hemarthrosis":
            injected_dose += (
                round(weight * constants.JOINT_BLEEDING_DOSE) * number_of_bleeds
            )
        case "lt_bleeding":
            injected_dose += round(weight * constants.LT_BLEEDING_DOSE)
        case "death":
            # to avoid assigning base prophylaxes dose to death
            injected_dose = 0

    return injected_dose


def on_demand_factor_consumption(
    step: int, state: str, number_of_bleeds: int, **psa_kwargs
):
    return factor_consumption(
        step, state, number_of_bleeds, strategy="on_demand", **psa_kwargs
    )


def prophylaxis_factor_consumption(
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
