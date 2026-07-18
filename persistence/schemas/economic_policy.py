from pydantic import BaseModel, Field
from typing import Literal


class GDPPerCapita(BaseModel):
    USD: float
    IRR: float
    TOMAN: float


class WTPMultiplier(BaseModel):
    standard: float
    rare: float


class EconomicPolicyFile(BaseModel):
    currency: Literal["USD", "IRR", "T"]
    disease_profile: Literal["standard", "rare"]
    gdp_per_capita: GDPPerCapita
    wtp_multiplier: WTPMultiplier
