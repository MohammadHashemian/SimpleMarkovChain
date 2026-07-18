import numpy as np

from engine.transitions import HybridTransitionGenerator


def assert_row_stochastic(P, tol=1e-10):
    for row in P:
        assert np.isclose(sum(row), 1.0, atol=tol)


def assert_valid_probabilities(P):
    for row in P:
        assert np.all(np.array(row) >= -1e-12)
        assert np.all(np.array(row) <= 1.0 + 1e-12)


def test_ctmc_rate_to_probability_valid_structure():
    states = [
        "healthy",
        "bleeding",
        "joint_bleeding",
        "death",
    ]

    hazards = {
        ("healthy", "bleeding"): (6.0, "weekly"),
        ("healthy", "joint_bleeding"): (20.0, "weekly"),
        ("bleeding", "death"): (2.0, "weekly"),
        ("joint_bleeding", "death"): (3.0, "weekly"),
    }

    tg = HybridTransitionGenerator(
        states=states,
        transition_pairs=hazards,
        time_step="weekly",
    )

    P = np.array(tg.build())

    assert_row_stochastic(P)
    assert_valid_probabilities(P)

    i = tg.state_indices["healthy"]
    row = P[i]

    p_bleeding = row[tg.state_indices["bleeding"]]
    p_joint = row[tg.state_indices["joint_bleeding"]]
    p_stay = row[i]

    assert p_joint > 0
    assert p_bleeding > 0
    assert p_stay > 0

    # hazard ordering preserved in transition mass
    assert p_joint > p_bleeding

    dt = 1.0 / 52.0

    Q_approx = (P - np.eye(len(states))) / dt

    q_row = Q_approx[i]

    q_bleeding = max(q_row[tg.state_indices["bleeding"]], 0.0)
    q_joint = max(q_row[tg.state_indices["joint_bleeding"]], 0.0)

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

    assert 0.0 < p_stay < 1.0


def test_death_is_absorbing():
    states = ["healthy", "death"]

    hazards = {("healthy", "death"): (1.0, "annual")}

    tg = HybridTransitionGenerator(states, hazards, time_step="annual")
    P = np.array(tg.build())

    d = tg.state_indices["death"]

    assert np.isclose(P[d, d], 1.0)
    assert np.allclose(P[d], [0.0, 1.0])
