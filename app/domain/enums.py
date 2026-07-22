from enum import StrEnum


class Regime(StrEnum):
    ON_DEMAND = "on_demand"
    PROPHYLAXIS = "prophylaxis"


class HealthStates(StrEnum):
    HEALTHY = "healthy"
    BLEEDING = "bleeding"
    HEMARTHROSIS = "hemarthrosis"
    LT_BLEEDING = "lt_bleeding"
    DEATH = "death"


class UtilityStates(StrEnum):
    HEALTHY = "healthy"
    MILD_ARTHROPATHY = "mild_arthropathy"
    MODERATE_ARTHROPATHY = "moderate_arthropathy"
    SEVERE_ARTHROPATHY = "severe_arthropathy"
    BLEEDING = "bleeding"
    HEMARTHROSIS = "hemarthrosis"
    LT_BLEEDING = "lt_bleeding"
    DEATH = "death"


class ArthropathySeverity(StrEnum):
    HEALTHY = "healthy"
    MILD = "mild_arthropathy"
    MODERATE = "moderate_arthropathy"
    SEVERE = "severe_arthropathy"
