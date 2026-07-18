from engine.transitions import HybridTransitionGenerator
import numpy as np


# helpers


def assert_row_stochastic(P, tol=1e-10):
    for row in P:
        assert np.isclose(sum(row), 1.0, atol=tol)


def assert_valid_probabilities(P):
    for row in P:
        assert np.all(np.array(row) >= -1e-12)  # allow tiny numerical noise
        assert np.all(np.array(row) <= 1.0 + 1e-12)


# MAIN CTMC TEST


def test_ctmc_rate_to_probability_valid_structure():
    """
    Validates CTMC correctness using generator-consistent invariants.
    """

    states = [
        "healthy",
        "bleeding",
        "joint_bleeding",
        "death",
    ]

    # annual hazard rates (λ)
    hazards = {
        ("healthy", "bleeding"): 6.0,
        ("healthy", "joint_bleeding"): 20.0,
        ("bleeding", "death"): 2.0,
        ("joint_bleeding", "death"): 3.0,
    }

    tg = HybridTransitionGenerator(
        states=states,
        transition_pairs=hazards,
        time_step="weekly",
    )

    P = np.array(tg.build())

    # 1. Markov validity
    assert_row_stochastic(P)
    assert_valid_probabilities(P)

    i = tg.state_index["HEALTHY"]
    row = P[i]

    p_bleeding = row[tg.state_index["BLEEDING"]]
    p_joint = row[tg.state_index["JOINT_BLEEDING"]]
    p_death = row[tg.state_index["DEATH"]]
    p_stay = row[i]

    # 2. STRUCTURAL invariant (NOT probability equality)
    # Important: for finite Δt, probabilities are nonlinear,
    # but ordering and dominance MUST hold

    assert p_joint > 0
    assert p_bleeding > 0
    assert p_stay > 0

    # hazard ordering preserved in transition intensity
    assert p_joint > p_bleeding  # 20 > 6 must reflect in transition mass

    # 3. generator reconstruction (correct CTMC test)
    dt = tg.delta_t

    Q_approx = (P - np.eye(len(states))) / dt

    q_row = Q_approx[i]

    q_bleeding = max(q_row[tg.state_index["BLEEDING"]], 0.0)
    q_joint = max(q_row[tg.state_index["JOINT_BLEEDING"]], 0.0)

    total_q = q_bleeding + q_joint

    if total_q > 0:
        assert np.isclose(
            q_bleeding / total_q,
            6.0 / (6.0 + 20.0),
            rtol=1e-2,
            atol=1e-2,
        )

        assert np.isclose(
            q_joint / total_q,
            20.0 / (6.0 + 20.0),
            rtol=1e-2,
            atol=1e-2,
        )

    # 4. survival sanity
    assert 0.0 < p_stay < 1.0


# absorbing state test


def test_death_is_absorbing():
    states = ["healthy", "death"]

    hazards = {("healthy", "death"): 1.0}

    tg = HybridTransitionGenerator(states, hazards, time_step="annual")
    P = np.array(tg.build())

    d = tg.state_index["DEATH"]

    assert np.isclose(P[d, d], 1.0)
    assert np.allclose(P[d], [0.0, 1.0])
