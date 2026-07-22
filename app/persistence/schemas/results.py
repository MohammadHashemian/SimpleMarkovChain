# Pydantic
from pydantic import BaseModel, ConfigDict

from engine.results import MarkovResult  # Generic model output


class HemophiliaOutput(BaseModel):
    """
    Specified markov model simulation results
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Composition
    simulation: MarkovResult

    # Hemophilia-specific fields
    factor_consumption: float
    factor_costs: float
    annual_factor_consumption: float
    annual_factor_costs: float
    hemarthrosis: float
    qaly: float
    abr: float
    pettersson_score: list[int]
