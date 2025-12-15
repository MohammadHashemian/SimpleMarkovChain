from typing import TypedDict, Optional, Union
from enum import StrEnum
from model.markov_chain import MarkovResult
from dataclasses import dataclass
import numpy as np

# dtype -> Defined Types


class Treatment(StrEnum):
    ON_DEMAND = "on_demand"
    PROPHYLAXIS = "prophylaxis"


class HemophiliaInput(TypedDict):
    treatment: Treatment
    abr: Union[np.float64, float]
    ajbr: Optional[Union[np.float64, float]]
    wbr: Optional[Union[np.float64, float]]
    wjbr: Optional[Union[np.float64, float]]
    webr: Optional[Union[np.float64, float]]


class HemophiliaRewardArgs(HemophiliaInput):
    """
    Summary
    -------
    Extends Hemophilia Input dictionary to type the reward keyword arguments

    Note:
    -------
    Do not forget to add new store shared reward states name to this typed dictionary
    """

    number_of_bleeds: int | None
    number_of_hemarthrosis: int | None
    pettersson_score: int | None


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


# TODO:
# Should i use dataclasses instead of typed dictionary? to get ride of Key Errors


@dataclass
class HemophiliaInputs:
    treatment: Treatment
    abr: np.float64 | float
    ajbr: Optional[Union[np.float64, float]] = None
    wbr: Optional[Union[np.float64, float]] = None
    wjbr: Optional[Union[np.float64, float]] = None
    webr: Optional[Union[np.float64, float]] = None


@dataclass
class HemophiliaOutputs:
    number_of_bleeds: int = 0
    number_of_hemarthrosis: int = 0
    pettersson_score = 0
