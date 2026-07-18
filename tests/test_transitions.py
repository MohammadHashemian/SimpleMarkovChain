import numpy as np
import pytest

from engine.transitions import (
    HybridTransitionGenerator,
    CTMCTransitionGenerator,
    IndependentHazardTransitionGenerator,
)


def assert_row_stochastic(P, rtol=1e-6):
    for row in P:
        assert np.isclose(sum(row), 1.0, rtol=rtol)


# ── HybridTransitionGenerator ──────────────────────────────────────
# API: transition_pairs = {(from,to): (value, period)}
#      period=None → direct prob, "weekly"/"annual" → hazard
#      special_transitions = {state: [prob, ...]}  (list, not dict)


class TestHybridTransitionGeneratorValidation:
    def test_empty_states_raises(self):
        with pytest.raises(ValueError, match="States list cannot be empty"):
            HybridTransitionGenerator(states=[], transition_pairs={})

    def test_invalid_state_pair_raises(self):
        with pytest.raises(ValueError, match="Invalid state pair"):
            HybridTransitionGenerator(
                states=["A", "B"],
                transition_pairs={("A", "C"): (0.1, None)},
            )

    def test_negative_value_is_clamped_to_zero(self):
        tg = HybridTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): (-0.5, "weekly")},
        )
        matrix = tg.build()
        assert matrix[0][1] >= 0.0  # no negative probabilities

    def test_special_transition_invalid_state(self):
        with pytest.raises(ValueError, match="not in states"):
            HybridTransitionGenerator(
                states=["A", "B"],
                transition_pairs={("A", "B"): (0.1, None)},
                special_transitions={"C": [0.5, 0.5]},
            )

    def test_special_transition_row_sum(self):
        with pytest.raises(ValueError, match="must sum to 1"):
            HybridTransitionGenerator(
                states=["A", "B"],
                transition_pairs={},
                special_transitions={"A": [0.5, 0.4]},
            )


class TestHybridTransitionGeneratorBuild:
    def test_basic_matrix_is_stochastic(self):
        tg = HybridTransitionGenerator(
            states=["A", "B", "C"],
            transition_pairs={
                ("A", "B"): (0.5, None),
                ("A", "C"): (0.3, None),
                ("B", "C"): (0.2, None),
            },
        )
        P = np.array(tg.build())
        assert_row_stochastic(P)

    def test_absorbing_state_when_no_outgoing(self):
        tg = HybridTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): (0.0, None)},
        )
        P = np.array(tg.build())
        assert np.isclose(P[1, 1], 1.0)
        assert_row_stochastic(P)

    def test_special_transitions_override(self):
        tg = HybridTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): (10.0, "weekly")},
            special_transitions={"A": [0.2, 0.8]},
        )
        P = np.array(tg.build())
        assert np.allclose(P[0], [0.2, 0.8])
        assert_row_stochastic(P)

    def test_absorbing_special_state(self):
        tg = HybridTransitionGenerator(
            states=["A", "B"],
            transition_pairs={},
            special_transitions={"A": [1.0, 0.0]},
        )
        P = np.array(tg.build())
        assert np.allclose(P[0], [1.0, 0.0])
        assert P[1, 1] == 1.0

    def test_get_probability(self):
        tg = HybridTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): (1.0, "weekly")},
        )
        p = tg.get_probability(1.0, "weekly")
        assert 0.0 < p < 1.0

    def test_identity_self_loop_when_no_transition_defined(self):
        tg = HybridTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): (0.3, None)},
        )
        P = np.array(tg.build())
        assert P[0, 0] > 0

    def test_direct_probability(self):
        tg = HybridTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): (0.5, None)},
            time_step="weekly",
        )
        P = np.array(tg.build())
        assert np.isclose(P[0, 1], 0.5)
        assert_row_stochastic(P)

    def test_annual_rate_conversion(self):
        tg = HybridTransitionGenerator(
            states=["Healthy", "Dead"],
            transition_pairs={("Healthy", "Dead"): (1.0, "annual")},
            time_step="weekly",
        )
        P = np.array(tg.build())
        p_death = P[0, 1]
        assert 0.01 < p_death < 0.05
        assert_row_stochastic(P)

    def test_competing_risks_sum_to_one(self):
        tg = HybridTransitionGenerator(
            states=["Healthy", "Bleeding", "Dead"],
            transition_pairs={
                ("Healthy", "Bleeding"): (5.0, "weekly"),
                ("Healthy", "Dead"): (0.5, "weekly"),
            },
            time_step="weekly",
        )
        P = np.array(tg.build())
        assert_row_stochastic(P)
        row = P[0]
        assert row[1] > 0
        assert row[2] > 0

    def test_build_matrix(self):
        tg = HybridTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): (0.3, None)},
        )
        matrix = tg.build_matrix()
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (2, 2)


# ── CTMCTransitionGenerator ────────────────────────────────────────
# API: transition_pairs = {(from,to): float}  (bare hazard rates)
#      special_transitions = {state: {target: prob}}  (nested dict)
#      No numpy_matrix() — use build_matrix()


class TestCTMCTransitionGenerator:
    def test_basic_stochastic(self):
        tg = CTMCTransitionGenerator(
            states=["healthy", "bleeding", "death"],
            transition_pairs={
                ("healthy", "bleeding"): 6.0,
                ("bleeding", "death"): 2.0,
            },
            time_step="weekly",
        )
        P = np.array(tg.build())
        assert_row_stochastic(P, rtol=1e-5)

    def test_valid_probability_range(self):
        tg = CTMCTransitionGenerator(
            states=["healthy", "death"],
            transition_pairs={("healthy", "death"): 1.0},
            time_step="annual",
        )
        P = np.array(tg.build())
        assert np.all(P >= -1e-12)
        assert np.all(P <= 1.0 + 1e-12)
        assert_row_stochastic(P, rtol=1e-5)

    def test_death_absorbing(self):
        tg = CTMCTransitionGenerator(
            states=["healthy", "death"],
            transition_pairs={("healthy", "death"): 1.0},
            time_step="annual",
        )
        P = np.array(tg.build())
        assert np.isclose(P[1, 1], 1.0)

    def test_competing_risks_preserves_order(self):
        tg = CTMCTransitionGenerator(
            states=["healthy", "bleeding", "joint_bleeding", "death"],
            transition_pairs={
                ("healthy", "bleeding"): 6.0,
                ("healthy", "joint_bleeding"): 20.0,
                ("bleeding", "death"): 2.0,
                ("joint_bleeding", "death"): 3.0,
            },
            time_step="weekly",
        )
        P = np.array(tg.build())
        i = tg.state_index["HEALTHY"]
        p_bleeding = P[i, tg.state_index["BLEEDING"]]
        p_joint = P[i, tg.state_index["JOINT_BLEEDING"]]
        assert p_joint > p_bleeding

    def test_no_transitions_means_stay(self):
        tg = CTMCTransitionGenerator(
            states=["A", "B"],
            transition_pairs={},
            time_step="weekly",
        )
        P = np.array(tg.build())
        assert np.allclose(P, np.eye(2))

    def test_build_matrix(self):
        tg = CTMCTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): 1.0},
            time_step="weekly",
        )
        matrix = tg.build_matrix()
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (2, 2)


# ── IndependentHazardTransitionGenerator ───────────────────────────
# API: same as CTMC: bare float hazards, nested dict special transitions


class TestIndependentHazardTransitionGenerator:
    def test_basic_stochastic(self):
        tg = IndependentHazardTransitionGenerator(
            states=["A", "B", "C"],
            transition_pairs={
                ("A", "B"): 0.5,
                ("A", "C"): 0.3,
                ("B", "C"): 0.2,
            },
        )
        P = np.array(tg.build())
        assert_row_stochastic(P)

    def test_independent_hazard_conversion(self):
        tg = IndependentHazardTransitionGenerator(
            states=["Healthy", "Dead"],
            transition_pairs={("Healthy", "Dead"): 1.0},
            time_step="weekly",
        )
        P = np.array(tg.build())
        p_death = P[0, 1]
        expected = 1 - np.exp(-1.0 / 52)
        assert np.isclose(p_death, expected, rtol=1e-6)

    def test_death_absorbing(self):
        tg = IndependentHazardTransitionGenerator(
            states=["healthy", "death"],
            transition_pairs={("healthy", "death"): 1.0},
            time_step="annual",
        )
        P = np.array(tg.build())
        assert np.isclose(P[1, 1], 1.0)

    def test_self_loop_stays(self):
        tg = IndependentHazardTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): 0.3},
        )
        P = np.array(tg.build())
        assert P[0, 0] > 0
        assert_row_stochastic(P)

    def test_build_matrix(self):
        tg = IndependentHazardTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): 1.0},
            time_step="weekly",
        )
        matrix = tg.build_matrix()
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (2, 2)
