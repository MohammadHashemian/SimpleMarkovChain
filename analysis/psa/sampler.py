import numpy as np

from analysis.psa.models import ParameterSet


class PSASampler:
    def __init__(self, param_set: ParameterSet, seed: int):
        self.param_set = param_set
        self.rng = np.random.default_rng(seed)

    def sample(self, n: int) -> dict[str, np.ndarray]:
        return {
            field: getattr(self.param_set, field).sample(n, self.rng)
            for field in self.param_set.__dataclass_fields__
        }

    def point(self) -> dict[str, float]:
        return {
            field: getattr(self.param_set, field).point()
            for field in self.param_set.__dataclass_fields__
        }
