from pydantic import BaseModel


class EventFractions(BaseModel):
    ajbr_fraction: float
    ltb_fraction: float


class LTBRate(BaseModel):
    on_demand: float
    prophylaxis: float


class EventRates(BaseModel):
    ltb_rate: LTBRate


class Epidemiology(BaseModel):
    event_fractions: EventFractions
    event_rates: EventRates


class UtilityDecrements(BaseModel):
    on_demand: float
    prophylaxis: float


class Utilities(BaseModel):
    decrements: UtilityDecrements


class StudyEstimate(BaseModel):
    mean: float
    sd: float
    size: float
    source: str | None = None
    doi: str | None = None


class ABREvidence(BaseModel):
    on_demand: list[StudyEstimate]
    prophylaxis: list[StudyEstimate]


class Evidence(BaseModel):
    abr: ABREvidence


class PetterssonThresholds(BaseModel):
    mild: int
    moderate: int
    max: int


class PetterssonScore(BaseModel):
    conversion_factor: float
    thresholds: PetterssonThresholds


class ClinicalScoring(BaseModel):
    pettersson_score: PetterssonScore


class Dosing(BaseModel):
    ir_prophylaxis_weekly_dose_ui: float
    standard_prophylaxis_weekly_dose_ui: float
    bleeding_dose_ui: float
    joint_bleeding_dose_ui: float
    lt_bleeding_dose_ui: float


class Treatment(BaseModel):
    dosing: Dosing


class ClinicalFile(BaseModel):
    epidemiology: Epidemiology
    utilities: Utilities
    clinical_scoring: ClinicalScoring
    treatment: Treatment
    evidence: Evidence
