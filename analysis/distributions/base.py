from typing import Protocol, TypedDict

import numpy as np


class Distribution(Protocol):
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray: ...

    def point(self) -> float: ...


class ConvergenceDiagnostics(TypedDict):
    r_hat: float
    ess: int
    divergences: int
    converged: bool
