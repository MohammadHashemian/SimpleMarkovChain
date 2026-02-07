from dataclasses import dataclass, replace
from model.constants import ModelConfig


@dataclass(frozen=True)
class ScenarioConfig:
    """Interface to override model parameters at runtime"""

    title: str
    n_cycles: int
    start_age: int
    discounting: bool


def build_config(base: ModelConfig, scenario: ScenarioConfig) -> ModelConfig:
    econ = base.economics
    if not scenario.discounting:
        econ = replace(
            econ, discount_rate_costs_annual=0.0, discount_rate_benefits_annual=0.0
        )
    sim = replace(base.simulation, n_cycles=scenario.n_cycles)
    return replace(base, simulation=sim, economics=econ)


def run_model(model, config):
    pass
