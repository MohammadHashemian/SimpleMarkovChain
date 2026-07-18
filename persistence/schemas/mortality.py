from pydantic import BaseModel


class MortalityFile(BaseModel):
    use_age_specific: bool
    crude_annual_rate: float
    age_specific: dict[str, float]
