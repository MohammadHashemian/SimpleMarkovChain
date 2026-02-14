from typing import Optional, Union
from enum import StrEnum
from dataclasses import dataclass, asdict
from model.markov_chain import MarkovResult
from model.config import ModelConfig
import numpy as np


class Regime(StrEnum):
    ON_DEMAND = "on_demand"
    PROPHYLAXIS = "prophylaxis"


@dataclass
class ModelInputAbs:
    """Common interface for model inputs, to be extended by specific model input dataclasses"""

    config: ModelConfig


@dataclass
class HemophiliaInput(ModelInputAbs):
    treatment: Regime
    abr: np.float64 | float
    ajbr: Optional[Union[np.float64, float]] = None
    wbr: Optional[Union[np.float64, float]] = None
    wjbr: Optional[Union[np.float64, float]] = None
    webr: Optional[Union[np.float64, float]] = None

    def to_dict(self):
        """
        Excremental method to convert dataclass to dictionary with string values for better readability in logs and progress bars.
        losses config as object, use to_dictionary method to preserve config as object in the dictionary.
        """
        return {k: str(v) for k, v in asdict(self).items()}

    def to_dictionary(self):
        """Convert to dict while preserving config as object"""
        d = {
            "config": self.config,  # Keep as object
            "treatment": self.treatment,
            "abr": self.abr,
            "ajbr": self.ajbr,
            "wbr": self.wbr,
            "wjbr": self.wjbr,
            "webr": self.webr,
        }
        return d


# Pydantic
class HemophiliaOutput(MarkovResult):
    """
    Specified markov model simulation results
    """

    factor_consumption: float
    factor_costs: float
    annual_factor_consumption: float
    annual_factor_costs: float
    hemarthrosis: float
    qaly: float
    abr: float
    pettersson_score: list[int | float]


@dataclass
class HemophiliaRewardArgs(HemophiliaInput):
    """
    Summary
    -------
    Extends Hemophilia Input dictionary to type the reward keyword arguments

    Note:
    -------
    Do not forget to add new store shared reward states name to this typed dictionary
    """

    number_of_bleeds: int | None = None
    number_of_hemarthrosis: int | None = None
    pettersson_score: int | None = None
