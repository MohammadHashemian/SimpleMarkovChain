from model.utils import prob_at_least_one
from model.constants import WEEKS_OF_YEAR, AJBR_FRACTION, LTB_FRACTION
import numpy as np
import pytest


@pytest.fixture()
def abrs():
    abr = [i for i in range(0, 160, 1)]
    return abr


# Assume states: [Healthy -Transitioning into---> [Bleeding, Hemarthrosis, LT_Bleeding, Death]]
def test_probability(abrs):
    # annual values
    annual_abr: float = np.random.choice(abrs)
    annual_ajbr: float = annual_abr * AJBR_FRACTION
    annual_ltb: float = annual_abr * LTB_FRACTION
    annual_aebr: float = annual_abr - (annual_ajbr + annual_ltb)
    # weekly values
    weekly_abr: float = annual_abr / WEEKS_OF_YEAR
    weekly_ajbr: float = annual_ajbr / WEEKS_OF_YEAR
    weekly_aebr: float = annual_aebr / WEEKS_OF_YEAR
    weekly_ltb: float = annual_ltb / WEEKS_OF_YEAR
    
    from_state = "Healthy"
    to_states = {
        "Bleeding": (weekly_aebr, "weekly"),
        "Hemarthrosis": (weekly_ajbr, "weekly"),
        "LT_Bleeding": (weekly_ltb, "weekly"),
    }
    print(f"\n \n Annual Bleeding Rate: {annual_abr}")
    print(
        f"""
        Bleeding frequency within a week (lambda value) WBR, WJBR, WEBR:
        {round(weekly_abr, 2)}, {round(weekly_ajbr, 2)}, {round(weekly_aebr, 2)}
        """
    )
    abr_prob = prob_at_least_one(weekly_abr)
    ajbr_prob = prob_at_least_one(weekly_ajbr)
    aebr_prob = prob_at_least_one(weekly_aebr)
    print(
        f"""
        Probability of at_least_one_event poisson distribution: 
        ABR: {round(abr_prob, 4)}
        AJBR: {round(ajbr_prob, 4)}
        AEBR: {round(aebr_prob, 4)}
        """
    )
