
from pydantic import BaseModel, ConfigDict


class MarkovResult(BaseModel):
    """
    Standardized result container for Markov model simulations.

    This model is returned by the generic Markov engine and can be used
    across PSA runs, base-case analysis, and scenario comparisons.
    """

    model_config = ConfigDict(frozen=True)
    initial_state: str
    final_state: str
    steps: int
    path: list[str]
