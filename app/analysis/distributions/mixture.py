from collections.abc import Sequence
from typing import Any

import numpy as np


class MixtureOfStudies:
    def __init__(self, components: Sequence[Any]) -> None:
        self.components = components

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        k = len(self.components)

        samples = np.column_stack([comp.sample(n, rng) for comp in self.components])

        weights = rng.dirichlet(np.ones(k), size=n)

        return np.sum(weights * samples, axis=1)

    def point(self) -> float:
        return float(np.mean([c.point() for c in self.components]))


class DirichletMixture:
    def __init__(
        self,
        components: Sequence,
        alpha: float | np.ndarray | None = None,
    ) -> None:
        self.components = components
        self.k = len(components)

        if alpha is None:
            self.alpha = np.ones(self.k)
        elif isinstance(alpha, float):
            self.alpha = np.full(self.k, alpha)
        else:
            self.alpha = np.asarray(alpha)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        component_samples = np.column_stack([c.sample(n, rng) for c in self.components])

        weights = rng.dirichlet(self.alpha, size=n)

        return np.sum(weights * component_samples, axis=1)

    def point(self) -> float:
        w = self.alpha / np.sum(self.alpha)
        return float(sum(wi * c.point() for wi, c in zip(w, self.components)))
