import numpy as np
import pytest

from engine.transitions import HybridTransitionGenerator


# Helpers


def assert_row_stochastic(P, rtol=1e-6):
    for row in P:
        assert np.isclose(sum(row), 1.0, rtol=rtol)


# Constructor validation


def test_empty_states_raises():
    with pytest.raises(ValueError, match="States list cannot be empty"):
        HybridTransitionGenerator(states=[], transition_pairs={})


def test_invalid_input_type():
    with pytest.raises(ValueError, match="Only hazard input"):
        HybridTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): 0.1},
            input_type="probability", # type: ignore
        )


def test_invalid_state_pair_raises():
    with pytest.raises(ValueError, match="Invalid state pair"):
        HybridTransitionGenerator(
            states=["A", "B"], transition_pairs={("A", "C"): 0.1}  # C not in states
        )


def test_negative_hazard_raises():
    with pytest.raises(ValueError, match="Hazards must be non-negative"):
        HybridTransitionGenerator(states=["A", "B"], transition_pairs={("A", "B"): -0.5})


def test_special_transition_invalid_state():
    with pytest.raises(ValueError, match="Unknown state"):
        HybridTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): 0.1},
            special_transitions={"C": {"A": 1.0}},
        )


def test_special_transition_row_sum():
    with pytest.raises(ValueError, match="must sum to 1"):
        HybridTransitionGenerator(
            states=["A", "B"],
            transition_pairs={},
            special_transitions={"A": {"A": 0.5, "B": 0.4}},
        )


# Absorbing state behavior


def test_absorbing_state_when_no_outgoing():
    tg = HybridTransitionGenerator(states=["A", "B"], transition_pairs={("A", "B"): 0.0})

    P = np.array(tg.build())
    assert np.isclose(P[1, 1], 1.0)  # B should be absorbing
    assert_row_stochastic(P)


# Hazard → probability correctness (basic sanity)


def test_hazard_conversion_non_negative():
    tg = HybridTransitionGenerator(states=["A", "B"], transition_pairs={("A", "B"): 1.0})

    p = tg._hazard_to_prob(1.0)
    assert 0.0 < p < 1.0


def test_matrix_is_stochastic():
    tg = HybridTransitionGenerator(
        states=["A", "B", "C"],
        transition_pairs={
            ("A", "B"): 0.5,
            ("A", "C"): 0.3,
            ("B", "C"): 0.2,
        },
    )

    P = np.array(tg.build())
    assert_row_stochastic(P)


# Special transitions override logic


def test_special_transitions_override():
    tg = HybridTransitionGenerator(
        states=["A", "B"],
        transition_pairs={("A", "B"): 10.0},
        special_transitions={"A": {"A": 0.2, "B": 0.8}},
    )

    P = np.array(tg.build())

    # row A must exactly match special transition
    assert np.allclose(P[0], [0.2, 0.8])
    assert_row_stochastic(P)


# Deterministic absorbing case


def test_absorbing_special_state():
    tg = HybridTransitionGenerator(
        states=["A", "B"],
        transition_pairs={},
        special_transitions={
            "A": {"A": 1.0},
        },
    )

    P = np.array(tg.build())
    assert np.allclose(P[0], [1.0, 0.0])
    assert P[1, 1] == 1.0
