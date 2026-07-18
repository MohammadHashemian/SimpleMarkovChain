from pydantic import BaseModel, Field
from typing import Literal


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


class Time(BaseModel):
    weeks_per_year: int


class SimulationFile(BaseModel):
    environment: Environment
    discounting: Discounting
    psa: PSA
    time: Time

    @property
    def is_development(self) -> bool:
        return self.environment.mode == "development"

    @property
    def sample_size(self) -> int:
        return self.psa.sample_size(self.environment.mode)
