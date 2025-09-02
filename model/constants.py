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
WTP_THRESHOLD = 4_771.4 * 3  # USD


WEEKS_OF_YEAR = 52  # Maybe 52.14 (?) <---------------------------------
START_SIMULATION_AGE_IN_WEEK = 2 * WEEKS_OF_YEAR  # 2 Years old patients
LONG_TERM_CYCLE_COUNTS = 70 * WEEKS_OF_YEAR  # 70 years in weeks (2, 72)
SHORT_TERM_CYCLE_COUNTS = 10 * WEEKS_OF_YEAR  # 10 years in weeks (2, 12)

# CONSIDERABLE
AJBR_FRACTION = 0.75  # Percentage of joint bleeds from all bleed events
LTB_FRACTION = 0.045  # Percent of life threatening bleeds from all bleed events
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

PPP_CONVERSION_FACTOR = 117_170  # World Bank 2024 IRR/USD, PPP
# PPP_CONVERSION_FACTOR = 165_354  # Y_CHART | IMF 2025 IRR/USD, PPP

# Tuned to fit Manco-Johnson M et al. article on NEW ENGLAND journal:
# Radio graphic results indicate 93% and 81% had no joint damage
# Rate indicates frequency of Hemarthrosis causes permanent joint damage (lambda)
LAM_ARTHROPATHY = 0.0040 


# IGNORED FEATURES
# DISCOUNT_RATE_WEEKLY = ((1 + 0.035) ** (1 / 52)) - 1
DISCOUNT_RATE_WEEKLY = None
