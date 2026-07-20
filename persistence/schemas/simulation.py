from typing import Literal

from pydantic import BaseModel

MortalitySource = Literal["iran", "poland", "default"]


class Environment(BaseModel):
    mode: Literal["development", "production"]
    seed: int


class Discounting(BaseModel):
    enable: bool
    cost_rate_annual: float
    utility_rate_annual: float


class PSA(BaseModel):
    development: int
    production: int

    def sample_size(self, mode: str) -> int:
        return getattr(self, mode)


class Mortality(BaseModel):
    """Selects which mortality table the model loads at startup.

    ``"iran"``     -> ``data/mortality_iran.json``  (UN WPP 2024, Male, Iran)
    ``"poland"``   -> ``data/mortality.json``        (default placeholder)
    ``"default"``  -> ``data/mortality.json``        (alias for ``"poland"``)
    """

    source: MortalitySource = "iran"


class Time(BaseModel):
    weeks_per_year: int


class SimulationFile(BaseModel):
    environment: Environment
    discounting: Discounting
    psa: PSA
    mortality: Mortality = Mortality()
    time: Time

    @property
    def is_development(self) -> bool:
        return self.environment.mode == "development"

    @property
    def sample_size(self) -> int:
        return self.psa.sample_size(self.environment.mode)
