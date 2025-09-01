from model.utils import prob_at_least_one, poisson_mass_function
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
    to_states = {
        "Bleeding": (weekly_aebr, "weekly"),
        "Hemarthrosis": (weekly_ajbr, "weekly"),
        "LT_Bleeding": (weekly_ltb, "weekly"),
    }
    print(f"\n \n Annual Bleeding Rate: {annual_abr}")
    print(
        f"""
        Bleeding frequency within a week (λ) WBR, WJBR, WEBR, WLTB:
        {round(weekly_abr, 2)}, {round(weekly_ajbr, 2)}, {round(weekly_aebr, 2)}, {round(weekly_ltb, 2)}
        """
    )
    no_bleed_prob = float(poisson_mass_function(lam=weekly_abr, k=0))
    ajbr_prob = prob_at_least_one(weekly_ajbr)
    aebr_prob = prob_at_least_one(weekly_aebr)
    ltb_prob = prob_at_least_one(weekly_ltb)
    sum_of_poisson = aebr_prob + ajbr_prob + ltb_prob + no_bleed_prob
    aebr_norm = aebr_prob / sum_of_poisson
    ajbr_norm = ajbr_prob / sum_of_poisson
    ltb_norm = ltb_prob / sum_of_poisson
    no_bleed_norm = no_bleed_prob / sum_of_poisson
    sum_of_poisson_norms = ltb_norm + ajbr_norm + aebr_norm + no_bleed_prob
    print(
        f"""
        Probability of no_event from poisson distribution:
        Healthy: {round(no_bleed_prob, 4)}
        Probability of at_least_one_event poisson distribution:
        Bleeding: {round(aebr_prob, 4)}
        Hemarthrosis: {round(ajbr_prob, 4)}
        LT_Bleeding: {round(ltb_prob, 4)}
        SUM: {round(sum_of_poisson, 4)}
        
        Normalized values:
        Healthy: {round(no_bleed_norm, 4)}
        Bleeding: {round(aebr_norm, 4)}
        Hemarthrosis: {round(ajbr_norm, 4)}
        LT_Bleeding: {round(ltb_norm, 4)}
        SUM: {round(sum_of_poisson_norms, 4)}
        """
    )
    # Using survival
    # Hazard based conversion
    total_lam = np.sum([_[0] for _ in to_states.values()])
    total_hazard_lam = np.sum(
        [-np.log(1 - prob_at_least_one(_[0])) for _ in to_states.values()]
    )
    survival = np.exp(-total_lam)
    print(
        f"""
        Sum of λ (total events happens on a week): {round(total_lam, 4)}
        Sum of hazard based conversion λ: {round(total_hazard_lam, 4)}
        with given λ, survival probability: {round(survival, 4)}
        """
    )
    lam_dict = {key: value[0] for key, value in to_states.items()}
    probs = {"Healthy": survival}
    print(f"    Probability value to Healthy: {round(probs['Healthy'], 4)}")
    for to_state, lam in lam_dict.items():
        if lam:
            probs.update({to_state: ((lam) / total_lam) * (1 - survival)})
        else:
            probs.update({to_state: 0})
        print(f"    Probability value to {to_state}: {round(probs[to_state], 4)}")
    sum_of_survival = np.sum([val for val in probs.values() if val])
    print(f"    Sum of values: {sum_of_survival}")
    print(
        "\nResult:\n Survival/competing risk method gained accurate transition matrix for lambda greater than 1, which is crucial in our study."
    )
