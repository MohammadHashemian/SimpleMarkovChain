from typing import TypedDict
from enum import StrEnum
from model.markov_chain import MarkovResult


import numpy as np


class Treatment(StrEnum):
    ON_DEMAND = "on_demand"
    PROPHYLAXIS = "prophylaxis"


class HemophiliaInput(TypedDict):
    treatment: Treatment
    abr: np.float64 | float
    ajbr: np.float64 | float | None
    wbr: np.float64 | float | None
    wjbr: np.float64 | float | None
    webr: np.float64 | float | None


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
