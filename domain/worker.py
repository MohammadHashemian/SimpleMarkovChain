from dataclasses import dataclass
from domain.transitions import AgeBasedMortalityModifier, build_transition_matrix
from persistence.schemas.utilities import StateUtilities

# from visualization.visualization import visualize_matrix
from domain.rewards.hemophilia import (
    make_pettersson_score,
    event_count,
    weight,
    consumption,
    utility,
)
from domain.enums import Regime
from domain.inputs import ModelInput
from domain.scenario import Scenario
from engine.chains import Chain, MarkovChains
import numpy as np

from persistence.context import ModelContext


def setup_rewards(
    markov: MarkovChains, inputs: ModelInput, regime: Regime, factor: float
):
    """_summary_

    Args:
        chain (MarkovChains): _description_
        inputs (ModelInput): _description_
    """
    # required arguments passes to store and reward functions
    markov.worker_kwargs.update({"regime": regime, "inputs": inputs})
    # requirements for reward functions
    markov.add_store_function("weight", weight)
    markov.add_store_function("event_count", event_count)
    markov.add_store_function("pettersson_score", make_pettersson_score(factor=factor))
    # reward functions
    markov.add_reward_function(consumption)
    markov.add_reward_function(utility)


@dataclass(frozen=True)
class ModelOutput:
    cycles: int
    regime: Regime
    total_factor: int
    factor_consumption: list[float]
    total_qaly: int
    qalys: list[float]
    mean_weight: int
    pettersson_score: int
    absorbed_at: int
    sequence: list[str]
    event_count: list[int]


def build_output(markov: MarkovChains, sequences, rewards, absorbed_at):
    cycles = markov.steps

    factor_seq = rewards[consumption.__name__][:]
    utility_seq = rewards[utility.__name__][:]
    weight_seq = rewards[weight.__name__][:]
    event_seq = rewards[event_count.__name__][:]
    pettersson_seq = rewards["pettersson_score"][:]

    regime = markov.worker_kwargs.get("regime", None)
    if not regime or not isinstance(regime, Regime):
        raise ValueError("regime is not defined")

    factor_sum = np.sum(factor_seq)
    qaly_sum = np.sum(utility_seq)
    mean_weight = np.mean(weight_seq)
    pettersson_score = int(pettersson_seq[-1])

    return ModelOutput(
        cycles=cycles,
        regime=regime,
        total_factor=factor_sum,
        factor_consumption=factor_seq,
        total_qaly=qaly_sum,
        qalys=utility_seq,
        mean_weight=mean_weight,
        pettersson_score=pettersson_score,
        absorbed_at=absorbed_at,
        event_count=event_seq,
        sequence=sequences,
    )


def _init_worker(inputs: ModelInput, context: ModelContext):
    global _reward_constants
    ctx = context
    inp = inputs

    utils = StateUtilities(
        healthy=inp.healthy_utility,
        bleeding=inp.spontaneous_bleeding_utility,
        hemarthrosis=inp.joint_bleeding_utility,
        lt_bleeding=inp.life_threatening_bleeding_utility,
        death=inp.death_utility,
        mild_arthropathy=inp.mild_arthropathy_utility,
        moderate_arthropathy=inp.moderate_arthropathy_utility,
        severe_arthropathy=inp.severe_arthropathy_utility,
    )

    _reward_constants = {
        "lam_bleed": inp.spontaneous_bleeding_rate / 52,
        "lam_joint": inp.joint_bleeding_rate / 52,
        "weekly_discount": (
            (1 + inp.benefits_discount_rate) ** (1 / 52) - 1
            if inp.benefits_discount_rate
            else 0
        ),
        "baseline_age_weeks": inp.baseline_age * 52,
        # flatten deep structures
        "utilities": utils,
        "threshold_mild": ctx.clinical.clinical_scoring.pettersson_score.thresholds.mild,
        "threshold_moderate": ctx.clinical.clinical_scoring.pettersson_score.thresholds.moderate,
        "threshold_max": ctx.clinical.clinical_scoring.pettersson_score.thresholds.max,
        "conversion_factor": ctx.clinical.clinical_scoring.pettersson_score.conversion_factor,
    }


def worker_function(
    chain: Chain,
    inputs: ModelInput,
    scenario: Scenario,
    context: ModelContext,
    run_id: int = 0,
    worker_id: int = 0,
):
    """
    Single markov model initializer function

    - Build transition matrix
    - Configure markov chain
    - Execute simulation
    - Aggregate outputs
    """
    # 0. Precompute constants
    _init_worker(inputs, context)

    # 1. Build transitions
    matrix = build_transition_matrix(inputs, chain.states)
    chain.update(matrix)

    # # DEBUG: Visualization
    # visualize_matrix(
    #     matrix=matrix,
    #     states=chain.states,
    #     inputs=inputs,
    #     sub=str(run_id),
    #     filename=f"matrix_{scenario.regime}_{worker_id}",
    # )

    # 2. Configure chain
    regime = scenario.regime
    chains = [chain]
    markov = MarkovChains(
        chains=chains,
        entrance="healthy",
        conditions=None,
        entrance_chain="main",
        steps=int(inputs.cycle),
        transition_modifier=AgeBasedMortalityModifier(mortality_file=context.mortality),
        worker_kwargs={},
        absorbing_states={"death"},
    )
    markov.worker_kwargs.update(
        {
            "const": _reward_constants,
            "rng": np.random.default_rng(
                context.simulation.environment.seed + worker_id
            ),  # Required for AgeBaseMortalityModifier
        }
    )
    setup_rewards(
        markov=markov,
        inputs=inputs,
        regime=regime,
        factor=_reward_constants["conversion_factor"],
    )

    # 3. Run simulation
    sequences = markov.run()
    absorbed_at = markov.absorbed_at
    rewards = markov.collect_rewards()

    # # 4. Build outputs
    outputs = build_output(markov, sequences, rewards, absorbed_at)

    return outputs
