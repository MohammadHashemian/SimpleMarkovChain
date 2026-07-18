from typing import Protocol, TypedDict

import numpy as np


class ConvergenceDiagnostics(TypedDict):
    r_hat: float
    ess: int
    divergences: int
    converged: bool


class Distribution(Protocol):
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray: ...

    def point(self) -> float: ...


class DiagnosticsProtocol(Protocol):
    def get_distribution_stats(self) -> dict[str, float]: ...

    def sample_study_level(
        self, n: int, rng: np.random.Generator | None = None
    ) -> np.ndarray: ...

    def summary(self, var_names: list | None = None, **kwargs): ...

    def convergence_diagnostics(self) -> ConvergenceDiagnostics: ...
