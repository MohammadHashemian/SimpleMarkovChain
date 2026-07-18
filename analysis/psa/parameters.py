import numpy as np

from analysis.distributions import Distribution


class Parameter:
    def __init__(
        self, distribution: Distribution, cache: np.ndarray | None = None
    ) -> None:
        self.cache = cache
        self.distribution = distribution

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.cache is not None:
            return self.cache
        return self.distribution.sample(n, rng)

    def point(self) -> float:
        return self.distribution.point()
