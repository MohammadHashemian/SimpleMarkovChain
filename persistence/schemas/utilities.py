from pydantic import BaseModel


class StateUtilities(BaseModel):
    healthy: float
    mild_arthropathy: float
    moderate_arthropathy: float
    severe_arthropathy: float
    bleeding: float
    hemarthrosis: float
    lt_bleeding: float
    death: float


class EventDisutilities(BaseModel):
    severe_arthropathy_bleeding: float


class UtilityFile(BaseModel):
    state_utilities: StateUtilities
    event_disutilities: EventDisutilities
