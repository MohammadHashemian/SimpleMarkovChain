from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
from scipy.linalg import expm

from utils.math import prob_at_least_one


class HybridTransitionGenerator:
    """
    Flexible generator supporting mixed direct probabilities + hazard rates.
    Generates discrete-time transition probability matrices for Markov models
    with competing risks.

    This class supports:
        - Direct probabilities (period=None)
        - Continuous-time hazard rates converted using exponential survival
        - Explicit (special) transition rows for absorbing or custom states
        - Fully generic state names — no hardcoded 'Death' requirement
    """

    def __init__(
        self,
        states: list[str],
        transition_pairs: dict[
            tuple[str, str],
            tuple[float | str, Literal["weekly", "annual"] | None],
        ],
        special_transitions: dict[str, list[float]] | None = None,
        time_step: Literal["weekly", "annual"] = "weekly",
    ) -> None:
        """
        Initialize the TransitionGenerator.

        Parameters
        ----------
        states : List[str]
            Ordered list of all states in the model.
        transition_pairs : Dict[(from_state, to_state), (value, period)]
            Transition definitions.
            - period=None   → direct probability
            - period='weekly'|'annual' → hazard rate
        special_transitions : Optional[Dict[str, List[float]]], optional
            Explicit full probability rows for specific states (e.g. absorbing states).
        time_step : {'weekly', 'annual'}, default='weekly'
            Discrete time step of the output Markov chain.
        """
        self.states = [str(s) for s in states]  # ensure strings
        self.transition_pairs = transition_pairs
        self.special_transitions = special_transitions or {}
        self.time_step = time_step
        self.state_indices = {state: idx for idx, state in enumerate(self.states)}

        self._validate()

    def _validate(self) -> None:
        """Validate inputs."""
        if not self.states:
            raise ValueError("States list cannot be empty.")

        # Validate transition pairs
        for (from_state, to_state), (value, period) in self.transition_pairs.items():
            if from_state not in self.states or to_state not in self.states:
                raise ValueError(f"Invalid state pair: ({from_state}, {to_state})")
            if not isinstance(value, (int, float, str)):
                raise ValueError(
                    f"Invalid value type for transition {from_state}→{to_state}."
                )
            if period not in {None, "annual", "weekly"}:
                raise ValueError(
                    f"Invalid period '{period}' for {from_state}→{to_state}."
                )

        # Validate special transitions
        for state, probs in self.special_transitions.items():
            if state not in self.states:
                raise ValueError(f"Special transition state '{state}' not in states.")
            if len(probs) != len(self.states):
                n = len(self.states)
                raise ValueError(
                    f"Special transition for '{state}' must contain {n} probabilities."
                )
            if not np.allclose(sum(probs), 1.0, rtol=1e-5):
                raise ValueError(
                    f"Special transition probabilities for '{state}' must sum to 1.0 "
                    f"(got {sum(probs):.6f})."
                )

    def get_probability(
        self,
        value: float | int | str,
        period: Literal["weekly", "annual"] | None,
    ) -> float:
        """Convert rate to probability or return direct probability."""
        if period is None:
            return float(value)

        rate = float(value)
        if period == self.time_step:
            lam_ts = rate
        elif period == "annual" and self.time_step == "weekly":
            lam_ts = rate / 52.0
        elif period == "weekly" and self.time_step == "annual":
            lam_ts = rate * 52.0
        else:
            raise ValueError(
                f"Cannot convert period '{period}' to time_step '{self.time_step}'."
            )

        return prob_at_least_one(lam_ts)

    def build(self) -> list[list[float]]:
        """
        Construct the transition probability matrix.

        Returns
        -------
        List[List[float]]
            Square transition matrix (rows sum to 1).
        """
        n = len(self.states)
        matrix: list[list[float]] = [[0.0] * n for _ in range(n)]

        # 1. Insert special transitions
        for state, probs in self.special_transitions.items():
            idx = self.state_indices[state]
            matrix[idx] = probs[:]

        # 2. Build rows for remaining states
        for state in self.states:
            if state in self.special_transitions:
                continue

            idx = self.state_indices[state]
            row = [0.0] * n

            # Get all defined transitions from this state
            transitions = {
                to_state: (value, period)
                for (from_state, to_state), (
                    value,
                    period,
                ) in self.transition_pairs.items()
                if from_state == state
            }

            if not transitions:
                row[idx] = 1.0  # Naturally absorbing
                matrix[idx] = row
                continue

            # Split into direct probs and rates
            direct_probs: dict[str, float] = {}
            rate_transitions: dict[str, tuple[float | str, str]] = {}

            for to_state, (value, period) in transitions.items():
                if period is None:
                    direct_probs[to_state] = float(value)
                else:
                    rate_transitions[to_state] = (value, period)

            # Apply direct probabilities
            total_direct = sum(direct_probs.values())
            if total_direct > 1.0:
                raise ValueError(
                    f"Sum of direct probabilities from '{state}' exceeds 1.0 ({total_direct:.6f})."
                )

            for to_state, p in direct_probs.items():
                row[self.state_indices[to_state]] = p

            remaining = 1.0 - total_direct

            # Apply competing risks (rates)
            if rate_transitions and remaining > 0:
                lam_dict: dict[str, float] = {}
                for to_state, (value, period) in rate_transitions.items():
                    rate = float(value)
                    lam = (
                        rate
                        if period == self.time_step
                        else (
                            rate / 52.0
                            if period == "annual" and self.time_step == "weekly"
                            else (
                                rate * 52.0
                                if period == "weekly" and self.time_step == "annual"
                                else rate
                            )
                        )
                    )
                    lam_dict[to_state] = max(lam, 0.0)

                total_lam = sum(lam_dict.values())

                if np.isclose(total_lam, 0.0):
                    row[idx] += remaining
                else:
                    survival = np.exp(-total_lam)
                    transition_mass = remaining * (1.0 - survival)

                    for to_state, lam in lam_dict.items():
                        if total_lam > 0:
                            row[self.state_indices[to_state]] += (
                                lam / total_lam
                            ) * transition_mass

                    row[idx] += remaining * survival

            else:
                row[idx] += remaining

            # Final check
            if not np.allclose(sum(row), 1.0, rtol=1e-5):
                raise ValueError(
                    f"Row for state '{state}' does not sum to 1.0 (sum = {sum(row):.6f})."
                )

            matrix[idx] = row

        return matrix

    def build_matrix(self) -> np.ndarray:
        """Return transition matrix as numpy array (most convenient)."""
        return np.array(self.build(), dtype=np.float64)

    def __repr__(self) -> str:
        return f"TransitionGenerator(n_states={len(self.states)}, time_step={self.time_step})"


class CTMCTransitionGenerator:
    """
    Continuous-Time Markov Chain (CTMC) Transition Matrix Generator.

    It builds the infinitesimal generator matrix Q and computes the exact
    transition probabilities using the matrix exponential:

        P(Δt) = expm(Q * Δt)

    Advantages:
        - Exact competing risks handling
        - Consistent time scaling (weekly / annual)
        - No ad-hoc approximations or proportional allocation
        - True Markov semi-group property

    Use this version for publication or when maximum theoretical correctness is required.
    """

    def __init__(
        self,
        states: list[str],
        transition_pairs: Mapping[
            tuple[str, str], float
        ],  # All inputs are HAZARDS (rates)
        special_transitions: dict[str, dict[str, float]] | None = None,
        time_step: Literal["weekly", "annual"] = "weekly",
    ) -> None:
        """
        Initialize CTMC Transition Generator.

        Parameters
        ----------
        states : List[str]
            List of all states (order defines matrix indexing).
        transition_pairs : Mapping[(from_state, to_state), hazard_rate]
            Annual or weekly hazard rates (λ). All transitions here are treated as rates.
        special_transitions : Dict[str, Dict[str, float]], optional
            Explicit probability distributions for specific states
            (e.g. absorbing states or LT_Bleeding). These override hazard logic.
        time_step : {'weekly', 'annual'}, default='weekly'
            Time step of the discrete Markov chain.
        """
        if not states:
            raise ValueError("States list cannot be empty.")

        self.states = [s.strip().upper() for s in states]
        self.state_index = {state: i for i, state in enumerate(self.states)}
        self.time_step = time_step
        self.delta_t = 1.0 / 52.0 if time_step == "weekly" else 1.0

        # Store hazards
        self.transition_pairs = {
            (src.upper(), dst.upper()): float(rate)
            for (src, dst), rate in transition_pairs.items()
        }

        # Special transitions (given as probabilities)
        self.special_transitions = {
            state.upper(): {target.upper(): float(p) for target, p in row.items()}
            for state, row in (special_transitions or {}).items()
        }

        self._validate()

    def _validate(self) -> None:
        """Validate inputs."""
        for (src, dst), rate in self.transition_pairs.items():
            if src not in self.state_index or dst not in self.state_index:
                raise ValueError(f"Invalid transition pair: {src} → {dst}")
            if rate < 0:
                raise ValueError(f"Negative hazard rate for {src} → {dst}")

        for state, row in self.special_transitions.items():
            if state not in self.state_index:
                raise ValueError(f"Unknown state in special_transitions: {state}")
            if not np.isclose(sum(row.values()), 1.0, rtol=1e-5):
                raise ValueError(f"Special transition for {state} must sum to 1.0")

    def _build_generator_matrix(self) -> np.ndarray:
        """Build the infinitesimal generator matrix Q.

        Math:
            Q[i, j] = λ_{i→j}              (off-diagonal, hazard rates)
            Q[i, i] = -sum_{j != i} Q[i,j] (diagonal, ensures row-sum = 0)

        For special_transitions, the user gives probabilities p instead of
        hazards. We convert via the equivalent-constant-hazard formula:
            λ = -ln(1 - p) / Δt
        which holds when the per-step transition is governed by a single
        Poisson process with constant rate. For absorbing rows (p_ii = 1,
        all other p = 0) the off-diagonals are all 0 and Q[i, i] = 0.
        """
        n = len(self.states)
        Q = np.zeros((n, n), dtype=float)

        # Fill off-diagonal elements with hazard rates
        for (src, dst), lam in self.transition_pairs.items():
            i = self.state_index[src]
            j = self.state_index[dst]
            Q[i, j] += lam

        # Override with special transitions (convert probabilities to hazards)
        for state, prob_row in self.special_transitions.items():
            i = self.state_index[state]
            Q[i, :] = 0.0  # Clear any previous hazards

            # Determine if this is an absorbing row (p_ii = 1, all others 0).
            # For an absorbing state, all off-diagonals must be 0 and Q[i,i] = 0.
            self_prob = prob_row.get(state, 0.0)
            is_absorbing = np.isclose(self_prob, 1.0, rtol=1e-12)

            for target, prob in prob_row.items():
                j = self.state_index[target]
                if i != j and prob > 0.0:
                    # Convert probability to equivalent constant hazard.
                    # Guard against numerical issues at p -> 1.
                    one_minus_p = 1.0 - prob
                    if one_minus_p > 1e-15:
                        Q[i, j] = -np.log(one_minus_p) / self.delta_t
                    else:
                        Q[i, j] = 0.0

            # Enforce CTMC property: row sum = 0.
            # For absorbing rows, off-diag is 0 so Q[i,i] stays 0 (correct).
            # For non-absorbing rows, Q[i,i] = -sum(off-diag).
            off_diag_sum = 0.0 if is_absorbing else np.sum(Q[i, :])
            Q[i, i] = -off_diag_sum

        # Enforce CTMC property for non-special rows
        for i in range(n):
            if self.states[i] in self.special_transitions:
                continue
            Q[i, i] = -np.sum(Q[i, :])

        return Q

    def build(self) -> list[list[float]]:
        """
        Compute discrete transition matrix P(Δt) = exp(Q Δt).

        Returns
        -------
        List[List[float]]
            Transition probability matrix (rows sum to 1).
        """
        Q = self._build_generator_matrix()

        # Matrix exponential
        P = expm(Q * self.delta_t)

        # Numerical safety
        P = np.real_if_close(P)
        P = np.clip(P, 0.0, None)  # Remove tiny negative values

        # Renormalize (handles floating-point drift)
        row_sums = P.sum(axis=1, keepdims=True)
        P = np.divide(P, row_sums, where=row_sums > 0, out=np.zeros_like(P))

        return P.tolist()

    def build_matrix(self) -> np.ndarray:
        """Return transition matrix as numpy array (recommended)."""
        return np.array(self.build(), dtype=np.float64)

    def __repr__(self) -> str:
        return f"CTMCTransitionGenerator(n_states={len(self.states)}, time_step='{self.time_step}')"


class IndependentHazardTransitionGenerator:
    """
    Discrete-Time Markov Chain builder using independent hazard conversion.

    Core Philosophy:
    ---------------
    - All regular transitions are given as **hazard rates (λ)**.
    - Each hazard is independently converted to probability using:
        p_i = 1 - exp(-λ_i * Δt)
    - Survival probability (staying in state) = exp(-Σλ * Δt)
    - Final normalization is applied for numerical stability.

    This is a **common and intuitive** approach in health economic modeling,
    but it is an approximation (slightly overestimates transitions when many
    competing risks are present).

    Use this when:
    - prefer simple, interpretable hazard → probability conversion
    - want to keep compatibility with your old way of thinking
    - don't need the absolute highest mathematical rigor (CTMC expm)

    For maximum rigor, use `CTMCTransitionGenerator` with matrix exponential instead.
    """

    def __init__(
        self,
        states: list[str],
        transition_pairs: Mapping[tuple[str, str], float],  # Hazards only
        special_transitions: dict[str, dict[str, float]] | None = None,
        time_step: Literal["weekly", "annual"] = "weekly",
    ) -> None:
        """
        Initialize the generator.

        Parameters
        ----------
        states : List[str]
            Ordered list of states.
        transition_pairs : Mapping[(from_state, to_state), hazard_rate]
            All transitions defined here are treated as constant hazard rates (λ).
        special_transitions : Dict[str, Dict[str, float]], optional
            Explicit probability rows (e.g. for absorbing states or LT_Bleeding).
            These bypass the hazard logic.
        time_step : {'weekly', 'annual'}, default='weekly'
            Time resolution of the Markov chain.
        """
        if not states:
            raise ValueError("States list cannot be empty.")

        self.states = [s.strip().upper() for s in states]
        self.state_index = {state: i for i, state in enumerate(self.states)}
        self.time_step = time_step
        self.delta_t = 1.0 / 52.0 if time_step == "weekly" else 1.0

        # Convert to normalized internal representation
        self.transition_pairs = {
            (src.upper(), dst.upper()): float(rate)
            for (src, dst), rate in transition_pairs.items()
        }

        self.special_transitions: dict[str, dict[str, float]] = {
            state.upper(): {target.upper(): float(p) for target, p in row.items()}
            for state, row in (special_transitions or {}).items()
        }

        self._validate()

    def _validate(self) -> None:
        """Input validation."""
        for (src, dst), rate in self.transition_pairs.items():
            if src not in self.state_index or dst not in self.state_index:
                raise ValueError(f"Invalid transition pair: {src} → {dst}")
            if rate < 0:
                raise ValueError(f"Negative hazard rate detected: {src} → {dst}")

        for state, row in self.special_transitions.items():
            if state not in self.state_index:
                raise ValueError(f"Unknown state in special_transitions: {state}")
            for target in row:
                if target not in self.state_index:
                    raise ValueError(
                        f"Invalid target in special transition {state}: {target}"
                    )
            if not np.isclose(sum(row.values()), 1.0, rtol=1e-5):
                raise ValueError(f"Special transition for {state} must sum to 1.0")

    def _hazard_to_prob(self, lam: float) -> float:
        """Convert hazard rate to transition probability over Δt."""
        return 1.0 - np.exp(-lam * self.delta_t)

    def build(self) -> list[list[float]]:
        """
        Build the transition probability matrix.

        Returns
        -------
        List[List[float]]
            Row-stochastic transition matrix.
        """
        n = len(self.states)
        P = np.zeros((n, n), dtype=np.float64)

        # 1. Apply special transitions (direct probabilities)
        for state, row in self.special_transitions.items():
            i = self.state_index[state]
            for target, prob in row.items():
                j = self.state_index[target]
                P[i, j] = prob

        # 2. Build regular states using independent hazard conversion
        for state in self.states:
            i = self.state_index[state]

            if state in self.special_transitions:
                continue

            # Get outgoing hazards from this state
            outgoing = {
                j: rate
                for (src, dst), rate in self.transition_pairs.items()
                if src == state and (j := self.state_index[dst])
            }

            if not outgoing:
                P[i, i] = 1.0
                continue

            # Convert each hazard independently
            trans_probs = {
                j: self._hazard_to_prob(rate) for j, rate in outgoing.items()
            }

            # Survival probability
            total_hazard = sum(outgoing.values())
            survival = np.exp(-total_hazard * self.delta_t)

            # Assign transition probabilities
            for j, p in trans_probs.items():
                P[i, j] = p

            P[i, i] = survival

            # Final normalization (important due to independence assumption)
            row_sum = P[i].sum()
            if row_sum > 0:
                P[i] = P[i] / row_sum
            else:
                P[i, i] = 1.0

        return P.tolist()

    def build_matrix(self) -> np.ndarray:
        """Return as numpy array (recommended for simulation)."""
        return np.array(self.build(), dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"IndependentHazardTransitionGenerator("
            f"n_states={len(self.states)}, time_step={self.time_step})"
        )


class DTMCTransitionGenerator:
    """Pure Discrete-Time Markov Chain (DTMC) transition matrix builder.

    Core Philosophy:
    ---------------
    - All inputs are **direct transition probabilities** (no hazards, no rates).
    - Each row of the output matrix is a probability distribution over target
      states and must sum to 1.
    - Absorbing states are first-class (rows where one entry is 1 and the rest
      are 0).
    - This is the simplest of the four generators and is appropriate when the
      modeler already has per-step probabilities (e.g. from a published decision
      model, calibration, or expert elicitation).

    Math:
    -----
    Given a set of states S and a transition specification that fully describes
    a row-stochastic matrix P where:

        P[i, j] = Pr(X_{t+1} = s_j | X_t = s_i)
        sum_j P[i, j] = 1  for all i

    the generator simply validates and assembles this matrix. No exponentials,
    no time-scale conversions, no proportional-allocation heuristics.

    API is symmetric with :class:`CTMCTransitionGenerator` and
    :class:`IndependentHazardTransitionGenerator`, but the values are
    probabilities (0 <= p <= 1) instead of rates (λ >= 0).
    """

    def __init__(
        self,
        states: list[str],
        transition_pairs: Mapping[tuple[str, str], float],
        special_transitions: dict[str, dict[str, float]] | None = None,
        time_step: Literal["weekly", "annual"] = "weekly",
    ) -> None:
        """
        Initialize the DTMC Transition Generator.

        Parameters
        ----------
        states : List[str]
            Ordered list of states.
        transition_pairs : Mapping[(from_state, to_state), probability]
            Direct per-step transition probabilities. Each value must lie in
            [0, 1]. For every state, the union of these entries plus any
            self-loop probability must sum to 1.
        special_transitions : Dict[str, Dict[str, float]], optional
            Explicit full-row probability distributions for specific states
            (e.g. absorbing states, custom sinks). When provided for a state,
            these override the ``transition_pairs`` entries from that state.
        time_step : {'weekly', 'annual'}, default='weekly'
            Accepted for API symmetry with the other generators. **Has no
            effect** on a DTMC matrix because no rate conversion is performed.
        """
        if not states:
            raise ValueError("States list cannot be empty.")

        self.states = [s.strip().upper() for s in states]
        self.state_index = {state: i for i, state in enumerate(self.states)}
        self.time_step = time_step

        self.transition_pairs = {
            (src.upper(), dst.upper()): float(p)
            for (src, dst), p in transition_pairs.items()
        }

        self.special_transitions: dict[str, dict[str, float]] = {
            state.upper(): {target.upper(): float(p) for target, p in row.items()}
            for state, row in (special_transitions or {}).items()
        }

        self._validate()

    def _validate(self) -> None:
        """Input validation."""
        for (src, dst), p in self.transition_pairs.items():
            if src not in self.state_index or dst not in self.state_index:
                raise ValueError(f"Invalid transition pair: {src} → {dst}")
            if p < 0.0 or p > 1.0:
                raise ValueError(
                    f"Transition probability for {src} → {dst} must be in [0, 1] "
                    f"(got {p})."
                )

        # Per-state row-sum check for non-special states. A state that has any
        # outgoing transition_pairs entries must have those entries sum to 1.0
        # (a complete distribution including any self-loop the user specifies).
        for state in self.states:
            if state in self.special_transitions:
                continue
            row_sum = sum(
                p
                for (src, _dst), p in self.transition_pairs.items()
                if src == state
            )
            if row_sum == 0.0:
                # No outgoing transitions specified: treat as absorbing (self-loop = 1).
                # Allow only if the state has no entries at all.
                has_any = any(src == state for (src, _dst) in self.transition_pairs)
                if has_any:
                    raise ValueError(
                        f"Transitions from '{state}' sum to 0; probabilities "
                        f"must be in [0, 1]."
                    )
                continue
            if not np.isclose(row_sum, 1.0, rtol=1e-5):
                raise ValueError(
                    f"Transitions from '{state}' sum to {row_sum:.6f}, expected 1.0. "
                    f"Make sure to include a self-loop (state -> state) for staying "
                    f"in place."
                )

        for state, row in self.special_transitions.items():
            if state not in self.state_index:
                raise ValueError(f"Unknown state in special_transitions: {state}")
            for target in row:
                if target not in self.state_index:
                    raise ValueError(
                        f"Invalid target in special transition {state}: {target}"
                    )
            for target, p in row.items():
                if p < 0.0 or p > 1.0:
                    raise ValueError(
                        f"Special transition probability for {state} → {target} "
                        f"must be in [0, 1] (got {p})."
                    )
            if not np.isclose(sum(row.values()), 1.0, rtol=1e-5):
                raise ValueError(
                    f"Special transition for {state} must sum to 1.0 "
                    f"(got {sum(row.values()):.6f})"
                )

    def build(self) -> list[list[float]]:
        """Build the transition probability matrix.

        Returns
        -------
        List[List[float]]
            Row-stochastic transition matrix (rows sum to 1).
        """
        n = len(self.states)
        P = np.zeros((n, n), dtype=np.float64)

        # 1. Apply special transitions first (they override the regular map).
        for state, row in self.special_transitions.items():
            i = self.state_index[state]
            for target, p in row.items():
                j = self.state_index[target]
                P[i, j] = p

        # 2. Build rows for remaining states directly from transition_pairs.
        for state in self.states:
            i = self.state_index[state]

            if state in self.special_transitions:
                continue

            row_total = 0.0
            for (src, dst), p in self.transition_pairs.items():
                if src != state:
                    continue
                j = self.state_index[dst]
                P[i, j] = p
                row_total += p

            # Absorbing state: nothing was specified, default to self-loop = 1.
            if row_total == 0.0:
                P[i, i] = 1.0
                continue

            # Sanity check (defensive — _validate() should have caught this).
            if not np.isclose(row_total, 1.0, rtol=1e-5):
                raise ValueError(
                    f"Transition probabilities from '{state}' sum to "
                    f"{row_total:.6f}, expected 1.0. Make sure to include a "
                    f"self-loop (state -> state) for staying in place."
                )

        return P.tolist()

    def build_matrix(self) -> np.ndarray:
        """Return as numpy array (recommended for simulation)."""
        return np.array(self.build(), dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"DTMCTransitionGenerator("
            f"n_states={len(self.states)}, time_step='{self.time_step}')"
        )
