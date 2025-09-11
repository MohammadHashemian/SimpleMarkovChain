# SOLID VALUES
START_STATE = "Healthy"
PRIMARY_STATES = [
    "Healthy",
    "Bleeding",
    "Hemarthrosis",
    "Arthropathy",
    "LT_Bleeding",
    "Death",
]
SECONDARY_STATES = ["Arthropathy", "Bleeding", "Hemarthrosis", "LT_Bleeding", "Death"]
GDP_PER_CAPITA = 4_771.4  # USD
WTP_THRESHOLD = GDP_PER_CAPITA * 3  # USD


WOY = 52  # Weeks of year maybe 52.14 (?) <-------------------
LONG_TERM_CYCLE_COUNTS = 70 * WOY  # 70 years in weeks (2, 72)
SHORT_TERM_CYCLE_COUNTS = 10 * WOY  # 10 years in weeks (2, 12)
SHORT_SIMULATION_START_AGE_IN_WEEK = 2 * WOY  # 2 Years old patients
MORTALITY_RATE = 4.9 / 1000 # Annually over 1000 population 4.9 person dies
# TODO: To be considered (?)
LONG_SIMULATION_START_AGE_IN_WEEK = SHORT_SIMULATION_START_AGE_IN_WEEK + SHORT_TERM_CYCLE_COUNTS # -> Age 12

# CONSIDERABLE
AJBR_FRACTION = 0.75  # Percentage of joint bleeds from all bleed events
LTB_FRACTION = 0.045  # Percent of life threatening bleeds from all bleed events
OD_LTB_RATE = 255 / 100_000  # TODO
PRO_LTB_RATE = 166 / 100_000  # TODO
RIAL_USD_PRICE = 853_661  # TGJU IRR/USD
PRICE_PER_UI_FACTOR_VIII = 58_000  # IRR, FDA
IR_PROPHYLAXIS_WEEKLY_DOSE = 25 * 2  # IR Protocol
STANDARD_PROPHYLAXIS_WEEKLY_DOSE = 25 * 3  # IR Protocol


# Guideline averages or PSA (?)
BLEEDING_DOSE = 30 * 4
JOINT_BLEEDING_DOSE = 30 * 2
LT_BLEEDING_DOSE = 550

# TODO:
WILLINGNESS_TO_PAY_THRESHOLD_IRR = None
LATE_ARTHROPATHY = None

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
