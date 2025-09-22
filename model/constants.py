# SOLID VALUES
START_STATE = "Healthy"
STATES = [
    "Healthy",
    "Bleeding",
    "Hemarthrosis",
    "LT_Bleeding",
    "Death",
]
STATE_UTILITIES = {
    "Healthy": 0.915,
    "Mild_Arthropathy": 0.85,
    "Moderate_Arthropathy": 0.80,
    "Severe_Arthropathy": 0.75,
    "Arthropathy": 0.75,  # Placeholder
    "Bleeding": 0.60,
    "Hemarthrosis": 0.50,
    "LT_Bleeding": 0.25,
    "Death": 0.0,
}
PETTERSSON_CATEGORIES = {
    i: (
        "Healthy"
        if i == 0
        else (
            "Mild_Arthropathy"
            if i <= 5
            else "Moderate_Arthropathy"
            if i <= 28
            else "Severe_Arthropathy"
        )
    )
    for i in range(80)
}
DECREMENT_PER_BLEED = {
    "on_demand": 0.0003725,
    "prophylaxis": 0.0018,
}  # Placeholder values
ON_DEMAND_ABR_REPORTS = [
    [58.3, 26.9],  # Zhao et al.
    [37.2, 19.9],  # Manco-Johnson MJ et al.
    [19.5, 15.0],  # Tagliaferri A et al.
    [17.7, 11.7],  # Tagliaferri A et al.
    # [13, 0],  # Gringeri A et al.
    [13, 12],  # Gringeri A et al. synthetic
    [7.4, 9.5],  # Romanová G et al.
    [13.2, 12.43],  # Berntorp E et al.
    [14.0, 12.3],  # Khair K et al.
    [18.4, 14.2],  # Khair K et al.
    [15.8, 8.13],  # Khair K et al.
]
PROPHYLAXIS_ABR_REPORTS = [
    [2.5, 4.6],  # Zhao et al.
    [2.5, 4.7],  # Manco-Johnson MJ et al.
    [2.6, 2.2],  # Tagliaferri A et al.
    [4.5, 7.1],  # Tagliaferri A et al.
    [6.3, 0],  # Gringeri A et al.
    [2.1, 2.1],  # Romanová G et al.
    [4.26, 5.97],  # Berntorp E et al.
    [3.5, 4.3],  # Khair K et al.
    [3.3, 4.1],  # Khair K et al.
    [3.7, 3.9],  # Khair K et al.
]
GDP_PER_CAPITA = 4_771.4  # USD
WTP_THRESHOLD = GDP_PER_CAPITA * 3  # USD


WOY = 52  # Weeks of year maybe 52.14 (?) <-------------------
LONG_TERM_CYCLE_COUNTS = 70 * WOY  # 70 years in weeks (2, 72)
SHORT_TERM_CYCLE_COUNTS = 10 * WOY  # 10 years in weeks (2, 12)
SHORT_SIMULATION_START_AGE_IN_WEEK = 2 * WOY  # 2 Years old patients
MORTALITY_RATE = 4.9 / 1000  # Annually over 1000 population 4.9 person dies
LONG_SIMULATION_START_AGE_IN_WEEK = (
    SHORT_SIMULATION_START_AGE_IN_WEEK + SHORT_TERM_CYCLE_COUNTS
)  # -> Age 12

# CONSIDERABLE
AJBR_FRACTION = 0.75  # Percentage of joint bleeds from all bleed events
LTB_FRACTION = 0.045  # Percent of life threatening bleeds from all bleed events
OD_LTB_RATE = 255 / 100_000  # TODO
PRO_LTB_RATE = 166 / 100_000  # TODO
RIAL_USD_PRICE = 853_661  # TGJU IRR/USD
PRICE_PER_UI_FACTOR_VIII = 58_000  # IRR, FDA
IR_PROPHYLAXIS_WEEKLY_DOSE = 25 * 2  # IR Protocol
STANDARD_PROPHYLAXIS_WEEKLY_DOSE = 25 * 3  # IR Protocol
PETTERSSON_CONVERSION_FACTOR = 12.6

# Guideline averages or PSA (?)
BLEEDING_DOSE = 30 * 4
JOINT_BLEEDING_DOSE = 30 * 2
LT_BLEEDING_DOSE = 550

# TODO:
WILLINGNESS_TO_PAY_THRESHOLD_IRR = None
LATE_ARTHROPATHY = None

REPORT_PPP = False
REPORT_UNIT = "USD"
PPP_CONVERSION_FACTOR = 117_170  # World Bank 2024 IRR/USD, PPP
# PPP_CONVERSION_FACTOR = 165_354  # Y_CHART | IMF 2025 IRR/USD, PPP

# Rate indicates frequency of Hemarthrosis causes permanent joint damage (lambda)
# Tuned to fit Manco-Johnson M et al. article on NEW ENGLAND journal:
# Radio graphic results indicate 93% and 81% had no joint damage
# MRI suggests 93% of prophylaxis and 55% of on demand had normal joint
EARLY_ARTHROPATHY = (
    0.05  # (PSA (?) as it's really effects the const-effectiveness results)
)

# IGNORED FEATURES (expenditures and effects are at same time, not needing to discount in this case)
# DISCOUNT_RATE_WEEKLY = ((1 + 0.035) ** (1 / 52)) - 1
DISCOUNT_RATE_WEEKLY = None
