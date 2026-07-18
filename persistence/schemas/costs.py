from typing import List
from pydantic import BaseModel


class Currency(BaseModel):
    name: str
    code: str
    symbol: str


class Assumption(BaseModel):
    iu_per_microgram: float


class Pricing(BaseModel):
    per_unit: dict[str, float]  # IRR, T, USD
    per_microgram: dict[str, float]  # IRR, T, USD


class CostItem(BaseModel):
    item: str
    assumption: Assumption
    pricing: Pricing


class CostFile(BaseModel):
    currencies: List[Currency]
    costs: List[CostItem]
