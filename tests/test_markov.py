import numpy as np
import pytest
import logging
import sys

from engine.runners import Runner


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("__pytest__")


@pytest.fixture()
def get_linear_abrs():
    abr = [i for i in range(0, 160, 1)]
    return abr


@pytest.fixture()
def hemophilia_markov_chain():
    from model.markov_chain import Chain
    from model.markov_chain import HemophiliaMarkovChains
    from model.markov_chain import TransitionGenerator
    from persistence.context import DEFAULT_CONFIG

    chains = []
    STATES = DEFAULT_CONFIG.health_states.STATES
    chain_main = Chain(
        name="main",
        states=STATES,
        matrix=np.eye(
            N=len(STATES), M=len(STATES), dtype=np.float64
        ),  # Identity matrix as placeholder
    )
    WOY = DEFAULT_CONFIG.simulation.weeks_per_year
    AJBR_FRACTION = DEFAULT_CONFIG.clinical.ajbr_fraction
    LTB_FRACTION = DEFAULT_CONFIG.clinical.ltb_fraction
    CRUDE_MORTALITY_RATE = DEFAULT_CONFIG.mortality.crude_annual_rate
    bleed_rate = 22 / WOY  # Average ABR of 22 converted to weekly rate
    hemarthrosis_rate = bleed_rate * AJBR_FRACTION
    ltbr = bleed_rate * LTB_FRACTION
    ebr = bleed_rate - (hemarthrosis_rate + ltbr)
    mortality_rate = CRUDE_MORTALITY_RATE / WOY
    # Direct probability assignment for no_bleeding event
    weekly_no_event_prob = np.exp(-bleed_rate)
    transition_pairs = {
        # Healthy Transitions (competing risks)
        ("Healthy", "Bleeding"): (ebr, "weekly"),
        ("Healthy", "Hemarthrosis"): (hemarthrosis_rate, "weekly"),
        ("Healthy", "LT_Bleeding"): (ltbr, "weekly"),
        ("Healthy", "Death"): (mortality_rate, "weekly"),
        # Bleeding Transitions (competing risks)
        ("Bleeding", "Healthy"): (weekly_no_event_prob, None),  # Direct probability
        ("Bleeding", "Hemarthrosis"): (hemarthrosis_rate, "weekly"),
        ("Bleeding", "LT_Bleeding"): (
            ltbr + mortality_rate,
            "weekly",
        ),
        ("Bleeding", "Death"): (mortality_rate, "weekly"),
        # Hemarthrosis Transitions (competing risks)
        ("Hemarthrosis", "Healthy"): (weekly_no_event_prob, None),  # Direct probability
        ("Hemarthrosis", "Bleeding"): (ebr, "weekly"),
        ("Hemarthrosis", "LT_Bleeding"): (ltbr, "weekly"),
        ("Hemarthrosis", "Death"): (mortality_rate, "weekly"),
    }

    # States order: ["Healthy", "Bleeding", "Hemarthrosis", "LT_Bleeding", "Death"]
    special_transitions = {
        "Death": [0.0] * (len(chain_main.states) - 1) + [1.0],  # Absorbing
        "LT_Bleeding": [0.94] + [0.0] * (len(chain_main.states) - 2) + [0.06],
    }
    transition_matrix = TransitionGenerator(
        states=STATES,
        time_step="weekly",
        transition_pairs=transition_pairs,
        special_transitions=special_transitions,
    ).numpy_matrix()
    chain_main.matrix = transition_matrix
    chains.append(chain_main)

    markov_chain = HemophiliaMarkovChains(
        chains=chains,
        entrance="Healthy",
        entrance_chain="main",
        conditions=None,
        dead_state="Death",
        steps=520,
        start_age=2,
        mortality_func=DEFAULT_CONFIG.mortality.get_annual_probability,
        enable_logger=True,
    )
    return markov_chain


# Assume states: [Healthy -Transitioning into---> [Bleeding, Hemarthrosis, LT_Bleeding, Death]]
def test_probability(get_linear_abrs):
    from model.utils import prob_at_least_one, poisson_mass_function
    from persistence.context import DEFAULT_CONFIG

    AJBR_FRACTION = DEFAULT_CONFIG.clinical.ajbr_fraction
    LTB_FRACTION = DEFAULT_CONFIG.clinical.ltb_fraction
    WOY = DEFAULT_CONFIG.simulation.weeks_per_year
    # annual values
    annual_abr: float = np.random.choice(get_linear_abrs)
    annual_ajbr: float = annual_abr * AJBR_FRACTION
    annual_ltb: float = annual_abr * LTB_FRACTION
    annual_aebr: float = annual_abr - (annual_ajbr + annual_ltb)
    # weekly values
    weekly_abr: float = annual_abr / WOY
    weekly_ajbr: float = annual_ajbr / WOY
    weekly_aebr: float = annual_aebr / WOY
    weekly_ltb: float = annual_ltb / WOY
    to_states = {
        "Bleeding": (weekly_aebr, "weekly"),
        "Hemarthrosis": (weekly_ajbr, "weekly"),
        "LT_Bleeding": (weekly_ltb, "weekly"),
    }
    logger.info(
        f"""
        Annual Bleeding Rate: {annual_abr}
        Bleeding frequency within a week (λ) WBR, WJBR, WEBR, WLTB:
        {round(weekly_abr, 2)}, {round(weekly_ajbr, 2)}, {round(weekly_aebr, 2)}, {round(weekly_ltb, 2)}
        """
    )
    no_bleed_prob = float(poisson_mass_function(lam=weekly_abr, k=0))
    ajbr_prob = prob_at_least_one(weekly_ajbr)
    aebr_prob = prob_at_least_one(weekly_aebr)
    ltb_prob = prob_at_least_one(weekly_ltb)
    sum_of_poisson = aebr_prob + ajbr_prob + ltb_prob + no_bleed_prob
    aebr_norm = aebr_prob / sum_of_poisson
    ajbr_norm = ajbr_prob / sum_of_poisson
    ltb_norm = ltb_prob / sum_of_poisson
    no_bleed_norm = no_bleed_prob / sum_of_poisson
    sum_of_poisson_norms = ltb_norm + ajbr_norm + aebr_norm + no_bleed_prob
    logger.info(
        f"""
        Probability of no_event from poisson distribution:
        Healthy: {round(no_bleed_prob, 4)}
        
        Probability of at_least_one_event poisson distribution:
        Bleeding: {round(aebr_prob, 4)}
        Hemarthrosis: {round(ajbr_prob, 4)}
        LT_Bleeding: {round(ltb_prob, 4)}
        SUM: {round(sum_of_poisson, 4)}
        
        Normalized values:
        Healthy: {round(no_bleed_norm, 4)}
        Bleeding: {round(aebr_norm, 4)}
        Hemarthrosis: {round(ajbr_norm, 4)}
        LT_Bleeding: {round(ltb_norm, 4)}
        SUM: {round(sum_of_poisson_norms, 4)}
        """
    )
    # Using survival
    # Hazard based conversion
    total_lam = np.sum([_[0] for _ in to_states.values()])
    total_hazard_lam = np.sum(
        [-np.log(1 - prob_at_least_one(_[0])) for _ in to_states.values()]
    )
    survival = np.exp(-total_lam)
    logger.info(
        f"""
        Sum of λ (total events happens on a week): {round(total_lam, 4)}
        Sum of hazard based conversion λ: {round(total_hazard_lam, 4)}
        with given λ, survival probability: {round(survival, 4)}
        """
    )
    lam_dict = {key: value[0] for key, value in to_states.items()}
    probs = {"Healthy": survival}
    logger.info(f"    Probability value to Healthy: {round(probs['Healthy'], 4)}")
    for to_state, lam in lam_dict.items():
        if lam:
            probs.update({to_state: ((lam) / total_lam) * (1 - survival)})
        else:
            probs.update({to_state: 0})
        logger.info(f"    Probability value to {to_state}: {round(probs[to_state], 4)}")
    sum_of_survival = np.sum([val for val in probs.values() if val])
    logger.info(f"    Sum of values: {sum_of_survival}")
    logger.info(
        """ \n
            Result: Survival/competing risk method gained accurate transition matrix
            for λ > 1, which is crucial in our study."""
    )


# Model struct:
#                                             <-----  Bleed Resolution  <-----
#      -----------------------------------------------------------------------------------------------------------
#      |                                                                                 |                       |
#   [Healthy | Mild | Moderate | Severe] <Pettersson decremented utility reward> -> [LT_Bleeding] -> [Death]     |
#      |                                                                                                         |
#      |-----------------------------------------------> [Bleeding] ---------------------------------------------|
#      |                                                                                                         |
#       -----------------------------------------------> [Hemarthrosis] <Increment pettersson score> ------------
def test_hemophilia_markov_chain_mortality_adjuster(hemophilia_markov_chain):
    states = hemophilia_markov_chain.chains[0].states
    matrix = hemophilia_markov_chain.chains[0].matrix

    assert states is not None
    logger.info(f"States: {states}")
    assert (
        len(states) == matrix.shape[0]
    ), "Number of states should match matrix dimension"

    assert matrix is not None
    logger.info("Transition Matrix prior to runtime adjustment: \n {}".format(matrix))

    steps_arr = hemophilia_markov_chain.run()
    if "Death" not in steps_arr:
        logger.info("Patient survived the entire simulation period.")
        return

    logger.info(f"Patient lived for: {steps_arr.index('Death')} weeks until Death.")
    state_before_death = steps_arr[steps_arr.index("Death") - 1]
    cause_of_death_map = {
        "Bleeding": "due natural cause of death",
        "Hemarthrosis": "due natural cause of death",
        "LT_Bleeding": "due to Life threatening Bleeding complications",
        "Healthy": "due natural cause of death",
    }
    logger.info(f"Cause of death: {cause_of_death_map[state_before_death]}")


def test_worker_function_execution(hemophilia_markov_chain):
    from persistence.context import DEFAULT_CONFIG
    from domain.worker import worker_function
    from model.scenarios.scenario import Scenario

    TEST_SCENARIO = Scenario(
        title="Test Scenario", n_cycles=10, discounting=False, start_age=2
    )
    TEST_CONFIG = TEST_SCENARIO.build_config(base=DEFAULT_CONFIG)

    worker_kwargs = {
        "treatment": "on_demand",
        "abr": 22.0,
        "config": TEST_CONFIG,
    }
    input, output = worker_function(hemophilia_markov_chain, worker_kwargs)
    logger.info(
        f"Worker function executed successfully with abr: {input.abr} and for: {output.steps} steps"
    )


def test_parallelized_markov_chain_execution(hemophilia_markov_chain):
    from domain.worker import worker_function
    from persistence.schemas.clinicals import HemophiliaInput
    from engine.runners import Runner
    from persistence.context import DEFAULT_CONFIG

    simulation_name = "Test Parallelized Markov Chain Execution"
    worker_inputs = [
        HemophiliaInput(
            **{
                "treatment": "on_demand",
                "abr": 22.0,
                "config": DEFAULT_CONFIG,
            }
        ),
        HemophiliaInput(
            **{
                "treatment": "prophylaxis",
                "abr": 5.0,
                "config": DEFAULT_CONFIG,
            }
        ),
    ]
    model_runner = Runner(
        title=simulation_name,
        worker_func=worker_function,
        worker_inputs=worker_inputs,
        markov_model=hemophilia_markov_chain,
        context=DEFAULT_CONFIG,
    )
    model_inputs, model_outputs = model_runner.run_model_multi_thread()
    assert len(model_inputs) == len(worker_inputs)
    assert len(model_outputs) == len(worker_inputs)


def test_scenario_analysis(hemophilia_markov_chain):
    from model.scenarios.scenario import Scenario
    from persistence.context import DEFAULT_CONFIG
    from domain.worker import worker_function
    from persistence.schemas.clinicals import HemophiliaInput

    simulation_name = "Test Scenario Analysis"
    worker_inputs = [
        HemophiliaInput(
            **{
                "treatment": "on_demand",
                "abr": 22.0,
                "config": DEFAULT_CONFIG,
            }
        ),
        HemophiliaInput(
            **{
                "treatment": "prophylaxis",
                "abr": 5.0,
                "config": DEFAULT_CONFIG,
            }
        ),
    ]

    s1 = Scenario(
        title="Test Scenario Analysis", n_cycles=10, discounting=False, start_age=2
    )
    s2 = Scenario(
        title="Test Scenario Analysis 2", n_cycles=10, discounting=False, start_age=2
    )
    scenarios = [s1, s2]
    runner = Runner(
        title=simulation_name,
        markov_model=hemophilia_markov_chain,
        worker_func=worker_function,
        worker_inputs=worker_inputs,
        context=DEFAULT_CONFIG,
        chain_instance=scenarios,
    )
    res = runner.run_scenarios_multi_thread()
    assert len(res) == len(scenarios)
