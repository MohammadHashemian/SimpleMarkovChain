import numpy as np
import pytest

from app.domain.modifiers import HemophiliaMortalityModifier
from engine.chains import (
    Chain,
    MarkovChains,
)
from engine.interfaces import NoOpModifier
from engine.results import MarkovResult
from engine.transitions import HybridTransitionGenerator

# Fixtures


@pytest.fixture
def simple_states():
    return ["Healthy", "Sick", "Dead"]


@pytest.fixture
def simple_transition_matrix():
    """Deterministic: Healthy -> Sick (100%), Sick -> Dead (100%), Dead absorbs"""
    return np.array(
        [
            [0.0, 1.0, 0.0],  # Healthy
            [0.0, 0.0, 1.0],  # Sick
            [0.0, 0.0, 1.0],  # Dead
        ]
    )


@pytest.fixture
def simple_chain(simple_states, simple_transition_matrix):
    return Chain(name="Standard", states=simple_states, matrix=simple_transition_matrix)


@pytest.fixture
def basic_markov(simple_chain):
    return MarkovChains(
        chains=[simple_chain],
        entrance="Healthy",
        entrance_chain="Standard",
        steps=5,
        transition_modifier=NoOpModifier(),
    )


@pytest.fixture
def mortality_func():
    """Simple step function: mortality rate per 1000 increases with age"""

    def func(age: int) -> float:
        if age < 30:
            return 1.0
        elif age < 60:
            return 5.0
        else:
            return 20.0

    return func


# Tests for TransitionGenerator
def test_transition_generator_builds_valid_matrix(simple_states):
    generator = HybridTransitionGenerator(
        states=simple_states,
        transition_pairs={
            ("Healthy", "Sick"): (0.1, None),  # direct prob
            ("Sick", "Dead"): (0.05, "annual"),  # rate
        },
        time_step="weekly",
    )
    matrix = generator.build_matrix()

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix.sum(axis=1), 1.0, rtol=1e-5)


# Absorbing states are optional
# def test_transition_generator_raises_on_invalid_states():
#     with pytest.raises(ValueError, match="Death state is required"):
#         TransitionGenerator(
#             states=["Healthy", "Sick"],
#             transition_pairs={},
#         )


# Tests for MarkovChains (generic)


def test_markov_chains_initialization(basic_markov):
    assert basic_markov.steps == 5
    assert basic_markov.entrance == "Healthy"
    assert basic_markov._current_chain_name == "Standard"


def test_markov_chains_run_returns_path(basic_markov):
    path = basic_markov.run()
    assert isinstance(path, list)
    assert len(path) > 0
    assert path[0] == "Healthy"


def test_markov_chains_collect_rewards(basic_markov):
    def dummy_reward(step: int, state: str, **kwargs):
        return step * 10 if state == "Healthy" else 0

    basic_markov.add_reward_function(dummy_reward)
    path = basic_markov.run()

    rewards = basic_markov.collect_rewards()
    assert "dummy_reward" in rewards
    assert len(rewards["dummy_reward"]) == len(path)


def test_markov_chains_stops_at_dead_state(simple_chain):
    model = MarkovChains(
        chains=[simple_chain],
        entrance="Healthy",
        entrance_chain="Standard",
        steps=10,
        transition_modifier=NoOpModifier(),
    )

    path = model.run()
    assert "Dead" in path
    # Should stop progressing after reaching Dead (no more state changes)
    assert path.count("Dead") >= 1


def test_markov_chains_with_store_function(basic_markov):
    def store_step(step: int, state: str, **kwargs):
        return step

    basic_markov.add_store_function("current_step", store_step)
    path = basic_markov.run()

    rewards = basic_markov.collect_rewards()
    assert "current_step" in rewards
    assert len(rewards["current_step"]) == len(path)


# Tests for Transition Modifiers


def test_noop_modifier_returns_copy(simple_chain):
    modifier = NoOpModifier()
    base_probs = simple_chain.matrix[0]  # from Healthy
    adjusted = modifier.adjust_transition(
        base_probs=base_probs,
        current_state="Healthy",
        current_chain_name="Standard",
        step=0,
        states=simple_chain.states,
    )
    assert np.array_equal(adjusted, base_probs)
    assert id(adjusted) != id(base_probs)  # should be a copy


def test_hemophilia_mortality_modifier_applies_on_yearly_boundary(
    simple_states, simple_transition_matrix, mortality_func
):
    modifier = HemophiliaMortalityModifier(
        mortality_func=mortality_func,
        start_age=25,
        dead_state="Dead",
        enable_logger=False,
    )

    base_probs = np.array([0.9, 0.1, 0.0])  # example probs from Healthy

    # Step 52 → end of first year (age 26)
    adjusted = modifier.adjust_transition(
        base_probs=base_probs,
        current_state="Healthy",
        current_chain_name="Standard",
        step=52,
        states=simple_states,
    )

    assert len(adjusted) == 3
    assert np.isclose(adjusted.sum(), 1.0, rtol=1e-8)
    assert adjusted[2] > base_probs[2]  # death probability increased


def test_hemophilia_mortality_modifier_skips_non_adjust_states(
    simple_states, simple_transition_matrix, mortality_func
):
    modifier = HemophiliaMortalityModifier(
        mortality_func=mortality_func,
        start_age=1,
        adjust_only_states=["Healthy"],
        enable_logger=False,
    )

    base_probs = np.array([0.8, 0.2, 0.0])

    adjusted = modifier.adjust_transition(
        base_probs=base_probs,
        current_state="Sick",  # not in adjust_only_states
        current_chain_name="Standard",
        step=52,
        states=simple_states,
    )

    assert np.array_equal(adjusted, base_probs)


def test_hemophilia_mortality_modifier_only_applies_yearly(
    simple_states, mortality_func
):
    modifier = HemophiliaMortalityModifier(
        mortality_func=mortality_func, start_age=1, enable_logger=False
    )

    base_probs = np.array([0.95, 0.05, 0.0])

    # Step 10 (not year boundary)
    adjusted = modifier.adjust_transition(
        base_probs=base_probs,
        current_state="Healthy",
        current_chain_name="Standard",
        step=10,
        states=simple_states,
    )

    assert np.allclose(adjusted, base_probs)


# Reproducibility & Edge Cases


def test_markov_chains_reproducible_with_seed(simple_chain):
    """Same seed → identical paths (when no stochastic modifier changes)"""
    np.random.seed(42)
    model1 = MarkovChains(
        chains=[simple_chain],
        entrance="Healthy",
        entrance_chain="Standard",
        steps=10,
        transition_modifier=NoOpModifier(),
    )
    path1 = model1.run()

    np.random.seed(42)
    model2 = MarkovChains(
        chains=[simple_chain],
        entrance="Healthy",
        entrance_chain="Standard",
        steps=10,
        transition_modifier=NoOpModifier(),
    )
    path2 = model2.run()

    assert path1 == path2


# ── Per-worker rng regression ────────────────────────────────────────
#
# Previously MarkovChains.walk drew transitions from the global
# np.random state, so the per-worker ``rng`` that worker_function
# passes via ``worker_kwargs['rng']`` was silently ignored. The
# following tests pin the contract that the engine MUST use the
# supplied Generator, never the global state.


def _stochastic_chain():
    """3-state chain with no absorbing state, so we exercise many draws."""
    states = ["A", "B", "C"]
    matrix = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.3, 0.5, 0.2],
            [0.2, 0.3, 0.5],
        ]
    )
    return Chain(name="main", states=states, matrix=matrix)


def test_markov_chains_uses_worker_kwargs_rng_when_provided():
    """Same per-worker rng seed → identical paths (the contract worker_function relies on)."""
    chain = _stochastic_chain()
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    p1 = MarkovChains(
        chains=[chain], entrance="A", entrance_chain="main", steps=100,
        transition_modifier=NoOpModifier(),
        worker_kwargs={"rng": rng_a},
    ).run()
    p2 = MarkovChains(
        chains=[chain], entrance="A", entrance_chain="main", steps=100,
        transition_modifier=NoOpModifier(),
        worker_kwargs={"rng": rng_b},
    ).run()
    assert p1 == p2


def test_markov_chains_different_worker_rng_seeds_give_different_paths():
    chain = _stochastic_chain()
    p42 = MarkovChains(
        chains=[chain], entrance="A", entrance_chain="main", steps=100,
        transition_modifier=NoOpModifier(),
        worker_kwargs={"rng": np.random.default_rng(42)},
    ).run()
    p7 = MarkovChains(
        chains=[chain], entrance="A", entrance_chain="main", steps=100,
        transition_modifier=NoOpModifier(),
        worker_kwargs={"rng": np.random.default_rng(7)},
    ).run()
    assert p42 != p7


def test_markov_chains_does_not_use_global_np_random_state():
    """Per-worker rng must override whatever the global state happens to be."""
    chain = _stochastic_chain()
    np.random.seed(123)  # dirty the global state
    p_dirty = MarkovChains(
        chains=[chain], entrance="A", entrance_chain="main", steps=100,
        transition_modifier=NoOpModifier(),
        worker_kwargs={"rng": np.random.default_rng(42)},
    ).run()
    np.random.seed(999)  # dirty the global state differently
    p_clean = MarkovChains(
        chains=[chain], entrance="A", entrance_chain="main", steps=100,
        transition_modifier=NoOpModifier(),
        worker_kwargs={"rng": np.random.default_rng(42)},
    ).run()
    # The per-worker rng is the same in both runs, so the paths must match
    # even though the global np.random state was changed between them.
    assert p_dirty == p_clean


def test_markov_chains_without_rng_isolated_from_global_state():
    """Even when no rng is supplied, the walk must not consume the global state."""
    chain = _stochastic_chain()
    np.random.seed(0)
    marker_before = np.random.get_state()[2]  # state position
    MarkovChains(
        chains=[chain], entrance="A", entrance_chain="main", steps=50,
        transition_modifier=NoOpModifier(),
    ).run()
    marker_after = np.random.get_state()[2]
    # Global state position should not have advanced.
    assert marker_before == marker_after


def test_markov_result_model():
    result = MarkovResult(
        initial_state="Healthy",
        final_state="Dead",
        steps=520,
        path=["Healthy"] * 100 + ["Dead"] * 421,
    )
    assert result.initial_state == "Healthy"
    assert result.steps == 520
