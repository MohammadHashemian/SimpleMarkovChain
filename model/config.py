from dataclasses import dataclass, field
from enum import StrEnum
from typing import Dict, List, Literal, Optional, Tuple


# TODO: Refactor the functions with new settings structure


class Currencies(StrEnum):
    IRR = "irr"
    TOMAN = "toman"
    USD = "usd"


class DevelopmentMode:
    ACTIVE: bool = True


@dataclass(frozen=True)
class SimulationSettings:
    """General simulation control parameters."""

    n_cycles: int = (
        98 * 52
    )  # Lifetime cycles (98 years × 52 weeks) Placeholder value, can be overridden by scenarios
    seed: int = 468498
    weeks_per_year: int = 52
    development_model: bool = DevelopmentMode.ACTIVE
    base_sobol_sample_size: int = field(
        default_factory=lambda: 64 if DevelopmentMode.ACTIVE else 512
    )  # Final Sobol sample size usually becomes N × (D + 2)


@dataclass(frozen=True)
class HealthStates:
    """All model health states + utility values."""

    START_STATE: Literal["Healthy"] = "Healthy"

    STATES: List[str] = field(
        default_factory=lambda: [
            "Healthy",
            "Bleeding",
            "Hemarthrosis",
            "LT_Bleeding",
            "Death",
        ]
    )

    utilities: Dict[str, float] = field(
        default_factory=lambda: {
            "Healthy": 0.915,
            "Mild_Arthropathy": 0.875,  # Mazza et al. 2016
            "Moderate_Arthropathy": 0.731,  # Mazza et al. 2016
            "Severe_Arthropathy": 0.54,  # Mazza et al. 2016 – consider PSA with Miners ~0.64
            "Bleeding": 0.60,
            "Hemarthrosis": 0.50,
            "LT_Bleeding": 0.25,
            "Death": 0.0,
        }
    )

    severe_arthropathy_bleeding_disutility: float = 0.10

    # Pettersson score → state mapping (0–78)
    pettersson_categories: Dict[int, str] = field(
        default_factory=lambda: {
            i: (
                "Healthy"
                if i == 0
                else (
                    "Mild_Arthropathy"
                    if i <= 4
                    else "Moderate_Arthropathy" if i <= 27 else "Severe_Arthropathy"
                )
            )
            for i in range(79)
        }
    )


@dataclass(frozen=True)
class Mortality:
    """Background mortality – age-specific or crude."""

    crude_annual_rate: float = 7.76 / 1000  # 2024 records per 1000 population

    @staticmethod
    def get_annual_probability(age: int, use_age_specific: bool = True) -> float:
        """
        Annual mortality probability (Poland life tables approximation).

        Source: ourworldindata.org / Poland records
        """
        if not use_age_specific:
            return Mortality.crude_annual_rate

        if age == 0:
            return 3.69 / 1000
        if 1 <= age <= 4:
            return 0.15 / 1000
        if 5 <= age <= 9:
            return 0.09 / 1000
        if 10 <= age <= 14:
            return 0.12 / 1000
        if 15 <= age <= 19:
            return 0.36 / 1000
        if 20 <= age <= 24:
            return 0.56 / 1000
        if 25 <= age <= 29:
            return 0.7 / 1000
        if 30 <= age <= 34:
            return 0.93 / 1000
        if 35 <= age <= 39:
            return 1.39 / 1000
        if 40 <= age <= 44:
            return 1.94 / 1000
        if 45 <= age <= 49:
            return 2.99 / 1000
        if 50 <= age <= 54:
            return 4.79 / 1000
        if 55 <= age <= 59:
            return 7.57 / 1000
        if 60 <= age <= 64:
            return 12.11 / 1000
        if 65 <= age <= 69:
            return 18.8 / 1000
        if 70 <= age <= 74:
            return 26.86 / 1000
        if 75 <= age <= 79:
            return 40.80 / 1000
        if 80 <= age <= 84:
            return 65.25 / 1000
        if 85 <= age <= 89:
            return 113.20 / 1000
        if age >= 90:
            return 200.0 / 1000  # capped

        return Mortality.crude_annual_rate  # fallback


@dataclass(frozen=True)
class EconomicParameters:
    """Currency, discounting, WTP, GDP settings."""

    currency: Currencies = Currencies.IRR

    discount_rate_costs_annual: float = 0.07
    discount_rate_benefits_annual: float = 0.03
    discount_rate_psa: Optional[float] = None  # can be set to None or different value

    @property
    def discount_rate_costs_weekly(self) -> float:
        return (1 + self.discount_rate_costs_annual) ** (1 / 52) - 1

    @property
    def discount_rate_benefits_weekly(self) -> float:
        return (1 + self.discount_rate_benefits_annual) ** (1 / 52) - 1

    # ── Base values (USD) ──
    gdp_per_capita_usd: float = 4_771.4
    wtp_multiplier_standard: int = 3
    report_ppp: bool = False
    ppp_conversion_factor: int = 117_170  # World Bank 2024 IRR/USD PPP

    # Overridden / computed values
    gdp_per_capita_local: float = field(init=False)
    wtp_threshold_local: float = field(init=False)

    def __post_init__(self):
        curr = self.currency.value.upper()

        if curr == "IRR":
            gdp = 180_000_000  # Toman equivalent
            wtp_mult = 10  # orphan/rare disease adjustment
        elif curr == "TOMAN":
            gdp = 180_000_000 / 10
            wtp_mult = 10
        else:
            gdp = self.gdp_per_capita_usd
            wtp_mult = self.wtp_multiplier_standard

        object.__setattr__(self, "gdp_per_capita_local", gdp)
        object.__setattr__(self, "wtp_threshold_local", gdp * wtp_mult)


@dataclass(frozen=True)
class ClinicalInputs:
    """ABR rates, arthropathy progression, bleeding fractions, doses etc."""

    # Annual bleeding rates – on-demand (mean, sd)
    on_demand_abr_reports: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (58.3, 26.9),  # Zhao et al. (median 12 y 1-50 y) 10.1177/1076029621989811
            (37.2, 19.9),  # Manco-Johnson MJ et al. (12-50) 10.1111/jth.13811
            (19.5, 15.0),  # Tagliaferri A et al. (12-25 y) 10.1160/TH14-05-0407
            (17.7, 11.7),  # Tagliaferri A et al. (26-55 y) 10.1160/TH14-05-0407
            (7.4, 9.5),  # Romanová G et al. (>=18 y, n=302) 10.1007/s00277-023-05453-6
            (16.8, 10.0),  # Belgium
            (13.8, 12.6),  # France
            (12.2, 18.1),  # Germany
            (11.5, 11.5),  # Italy
            (7.0, 6.4),  # Spain
            (4.5, 0.7),  # Sweden
            (19.4, 10.6),  # UK
            (
                14.0,
                12.3,
            ),  # Khair K et al. (median 17 y - n:299) 10.1111/hae.13361 - First year
            (18.4, 14.3),  # Khair K et al. - Second year
            (15.8, 8.13),  # Khair K et al. - Third year
            (58.9, 16.6),  # Zhao et al. (2-12 y, n=30) 10.1080/08880018.2017.1313921
            (5.6, 1.83),  # Eshghi et al. (<15 y, n=24) 10.1177/1076029616685429
            (13.9, 4.47),  # Roberto Musso (23.6 y, n=220) 10.1160/TH07-06-0409
            (57.7, 24.6),  # K. Kavakli (12-65 mean 28y) 10.1111/jth.12828
            (
                37.9,
                33.08,
            ),  # Fukutake (352 PTPs, 75.6% severe, 1-76 mean 25.8 y) 10.1007/s12185-018-02574-x
            (22.2, 7.0),  # Ying Liu (n=34; 4-18y, mean 12.2y) 10.1111/hae.14016
            (
                17.7,
                9.3,
            ),  # B Warren (n:37, 2.5 up to 7.5y) 10.1182/bloodadvances.2019001311
            (
                17.69,
                9.25,
            ),  # Marilyn J. Manco-Johnson (<1.5y to 6y, n:65) 10.1056/NEJMoa067659
            (
                13,
                9,
            ),  # Melissa Kern (n:15, pre target joint) 10.1016/j.jpeds.2004.06.082
            (
                24,
                12,
            ),  # Melissa Kern (n:15, post target joint) 10.1016/j.jpeds.2004.06.082
            (
                35.8,
                24.8,
            ),  # A. Tagliaferri (n: 83, median 23.6, 10-72y) 10.1111/j.1365-2516.2008.01791.x
            (21.42, 9.59),  # Aznar (n: 15, 26-47 mean 35.6) 10.1111/vox.12066
            (
                35.7,
                22.2,
            ),  # von Drygalski (26 adults, mean 42.8 y) 10.1056/NEJMoa2209226
            (22.2, 7.0),  # Liu, ODT group (n=18, age 12.4 SD 4.1) 10.1111/hae.14016
        ]
    )

    # Annual bleeding rates – prophylaxis (mean, sd)
    prophylaxis_abr_reports: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (2.5, 4.6),  # Zhao et al. (median 12 y 1-50 y) 10.1177/1076029621989811
            (2.5, 4.7),  # Manco-Johnson MJ et al. (12-50) 10.1111/jth.13811
            (2.6, 2.2),  # Tagliaferri A et al. (12-25 y) 10.1160/TH14-05-0407
            (4.5, 7.1),  # Tagliaferri A et al. (26-55 y) 10.1160/TH14-05-0407
            (2.1, 2.1),  # Romanová G et al. (>=18 y, n=302) 10.1007/s00277-023-05453-6
            (4.1, 6.9),  # Belgium
            (8.0, 9.4),  # France
            (4.5, 5.3),  # Germany
            (2.8, 4.7),  # Italy
            (2.7, 2.5),  # Spain
            (1.9, 2.9),  # Sweden
            (5.8, 7.0),  # UK
            (
                3.5,
                4.3,
            ),  # Khair K et al. (median 17 y - n:299) 10.1111/hae.13361 - First year
            (3.3, 4.1),  # Khair K et al. - Second year
            (3.7, 3.9),  # Khair K et al. - Third year
            (3.0, 5.9),  # Zhao et al. (2-12 y, n=30) 10.1080/08880018.2017.1313921
            (1.86, 1.52),  # Eshghi et al. (<15 y, n=24) 10.1177/1076029616685429
            (4.8, 5.0),  # Roberto Musso (23.6 y, n=220) 10.1160/TH07-06-0409
            (4.3, 6.5),  # K. Kavakli (12-65 mean 28y) 10.1111/jth.12828
            (
                8.9,
                19.61,
            ),  # Fukutake (352 PTPs, 75.6% severe, 1-76 mean 25.8 y) 10.1007/s12185-018-02574-x
            (
                3.5,
                2.1,
            ),  # Beth Boulden Warren (n:37, 2.5 up to 18y) - Early proph group 10.1182/bloodadvances.2019001311
            (
                6.2,
                5.3,
            ),  # Beth Boulden Warren - Post-proph group 10.1182/bloodadvances.2019001311
            (
                3.27,
                6.24,
            ),  # Marilyn J. Manco-Johnson (<1.5y to 6y, n:65) 10.1056/NEJMoa067659
            (
                4.2,
                3.7,
            ),  # A. Tagliaferri (n: 83, median 23.6, 10-72y) 10.1111/j.1365-2516.2008.01791.x
            (1.82, 2.87),  # Aznar (n: 15, 26-47 mean 35.6) 10.1111/vox.12066
            (3.2, 5.4),  # von Drygalski (133, ≥12 y, mean 33.9 y) 10.1056/NEJMoa2209226
        ]
    )

    decrement_per_bleed: Dict[str, float] = field(
        default_factory=lambda: {
            "on_demand": 0.0003725,
            "prophylaxis": 0.0018,
        }
    )

    # Bleeding fractions and rates
    ajbr_fraction: float = 0.75  # Percentage of joint bleeds from all bleed events
    # Percent of life threatening bleeds from all bleed events
    ltb_fraction: float = 0.045
    # On-demand LTB rate per 100k per year
    od_ltb_rate_per_100k: float = 255 / 100_000
    # Prophylaxis LTB rate per 100k per year
    pro_ltb_rate_per_100k: float = 166 / 100_000

    # Arthropathy progression
    # Lambda – probability per hemarthrosis (Rate indicates frequency of Hemarthrosis causes permanent joint damage)
    early_arthropathy_rate: float = 0.05
    # Factor to convert hemarthrosis count to Pettersson score
    pettersson_conversion_factor: float = 12.6

    # Economic parameters
    rial_usd_price: int = 853_661  # TGJU IRR/USD exchange rate
    # IRR per unit (IU) of Factor VIII, FDA standard
    price_per_ui_factor_viii_irr: int = 58_000

    # Treatment dosing (Intensity Regimen Protocol)
    # IR Protocol: 50 IU/kg twice weekly
    ir_prophylaxis_weekly_dose_ui: int = 25 * 2
    # Standard Protocol: 25 IU/kg three times weekly
    standard_prophylaxis_weekly_dose_ui: int = 25 * 3

    # Bleeding treatment doses (per bleeding event, before body weight adjustment)
    bleeding_dose_ui: int = 30 * 4  # Standard bleeding dose: 30 IU/kg × 4 injections
    joint_bleeding_dose_ui: int = 30 * 2  # Joint bleeding dose: 30 IU/kg × 2 injections
    lt_bleeding_dose_ui: int = 550  # Life-threatening bleeding dose: 550 IU/kg


@dataclass(frozen=True)
class ModelConfig:
    """Unified model config"""

    simulation: SimulationSettings = SimulationSettings()
    health_states: HealthStates = HealthStates()
    mortality: Mortality = Mortality()
    economics: EconomicParameters = EconomicParameters()
    clinical: ClinicalInputs = ClinicalInputs()


# ============================================================================
# Default global configuration instance
# ============================================================================
DEFAULT_CONFIG = ModelConfig()


# ============================================================================
# Pickle-friendly wrapper for mortality probability calculation
# ============================================================================
# This function is defined at module level to be pickle-safe for multiprocessing
def get_annual_mortality_probability(age: int, use_age_specific: bool = True) -> float:
    """
    Module-level wrapper for mortality probability calculation.

    This ensures the function can be properly pickled by multiprocessing,
    unlike bound methods on dataclass instances.

    Args:
        age: Age in years
        use_age_specific: If True, use age-specific rates; else use crude rate

    Returns:
        Annual mortality probability
    """
    return Mortality.get_annual_probability(age, use_age_specific)


# ============================================================================
# Legacy compatibility aliases - access through DEFAULT_CONFIG
# ============================================================================
def _get_config() -> ModelConfig:
    """Returns the default model configuration."""
    return DEFAULT_CONFIG


# Legacy constant accessors for backward compatibility
def get_woy() -> int:
    """Get weeks per year."""
    return _get_config().simulation.weeks_per_year


def get_wtp_threshold() -> float:
    """Get willingness-to-pay threshold."""
    return _get_config().economics.wtp_threshold_local


def get_model_currency() -> Currencies:
    """Get model currency."""
    return _get_config().economics.currency


def get_report_ppp() -> bool:
    """Get PPP reporting flag."""
    return _get_config().economics.report_ppp


def get_gdp_per_capita() -> float:
    """Get GDP per capita in local currency."""
    return _get_config().economics.gdp_per_capita_local


def get_discount_rate_weekly(rate_type: str = "costs") -> float:
    """Get weekly discount rate.

    Args:
        rate_type: Either 'costs' or 'benefits'
    """
    cfg = _get_config()
    if rate_type == "costs":
        return cfg.economics.discount_rate_costs_weekly
    elif rate_type == "benefits":
        return cfg.economics.discount_rate_benefits_weekly
    else:
        raise ValueError(f"Unknown rate type: {rate_type}")


# Aliases for direct import compatibility
WOY = 52  # Will use get_woy() instead
AJBR_FRACTION = DEFAULT_CONFIG.clinical.ajbr_fraction
LTB_FRACTION = DEFAULT_CONFIG.clinical.ltb_fraction
CRUDE_MORTALITY_RATE = DEFAULT_CONFIG.mortality.crude_annual_rate
MODEL_CURRENCY = DEFAULT_CONFIG.economics.currency
REPORT_PPP = DEFAULT_CONFIG.economics.report_ppp
PPP_CONVERSION_FACTOR = DEFAULT_CONFIG.economics.ppp_conversion_factor
RIAL_USD_PRICE = DEFAULT_CONFIG.clinical.rial_usd_price
PRICE_PER_UI_FACTOR_VIII = DEFAULT_CONFIG.clinical.price_per_ui_factor_viii_irr
STATE_UTILITIES = DEFAULT_CONFIG.health_states.utilities
PETTERSSON_CATEGORIES = DEFAULT_CONFIG.health_states.pettersson_categories
SEVERE_ARTHROPATHY_BLEEDING_DISUTILITY = (
    DEFAULT_CONFIG.health_states.severe_arthropathy_bleeding_disutility
)
STANDARD_PROPHYLAXIS_WEEKLY_DOSE = (
    DEFAULT_CONFIG.clinical.standard_prophylaxis_weekly_dose_ui
)
BLEEDING_DOSE = DEFAULT_CONFIG.clinical.bleeding_dose_ui
JOINT_BLEEDING_DOSE = DEFAULT_CONFIG.clinical.joint_bleeding_dose_ui
LT_BLEEDING_DOSE = DEFAULT_CONFIG.clinical.lt_bleeding_dose_ui
PETTERSSON_CONVERSION_FACTOR = DEFAULT_CONFIG.clinical.pettersson_conversion_factor

# Computed constants for scenario scenarios
PEDIATRIC_STARTING_POINT = 2 * WOY
PRIMARY_CYCLE_COUNT = 10 * WOY
LIFETIME_CYCLE_COUNTS = 98 * WOY
SECONDARY_CYCLE_COUNTS = 88 * WOY
ADOLESCENT_STARTING_POINT = PEDIATRIC_STARTING_POINT + PRIMARY_CYCLE_COUNT

# Dynamic discount rate properties
DISCOUNT_RATE_WEEKLY = DEFAULT_CONFIG.economics.discount_rate_costs_weekly
COSTS_DISCOUNT_RATE_WEEKLY = DEFAULT_CONFIG.economics.discount_rate_costs_weekly
BENEFITS_DISCOUNT_RATE_WEEKLY = DEFAULT_CONFIG.economics.discount_rate_benefits_annual
PSA_DISCOUNT_RATE_WEEKLY = DEFAULT_CONFIG.economics.discount_rate_psa

# WTP Threshold (computed from economics)
WTP_THRESHOLD = DEFAULT_CONFIG.economics.wtp_threshold_local

# Health states and utilities
START_STATE = DEFAULT_CONFIG.health_states.START_STATE
STATES = DEFAULT_CONFIG.health_states.STATES


# Legacy function kept for backward compatibility
def get_mortality_rate(age: int, use_age_specific: bool = True) -> float:
    """
    Return annual mortality probability for a given age.

    This delegates to the Mortality dataclass method.

    Reference: https://ourworldindata.org/grapher/annual-death-rate-by-age-group?country=~POL

    Parameters:
    - age: Current age in years (integer)
    - use_age_specific: If True, use realistic age-specific rates; if False, use constant crude rate

    Returns:
    - Annual mortality rate (float)
    """
    return Mortality.get_annual_probability(age, use_age_specific)
