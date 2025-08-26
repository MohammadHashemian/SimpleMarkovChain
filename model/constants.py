# Constants
LONG_TERM_CYCLE_COUNTS = 70 * 52  # 70 years in weeks
SHORT_TERM_CYCLE_COUNTS = 10 * 52 # 10 years in weeks
START_SIMULATION_AGE_IN_WEEK = 2 * 52 # 2 Years old patients
AJBR_FRACTION = 0.75 # Percentage of joint bleeds from all bleed events
LTB_FRACTION = 0.045 # Percent of life threatening bleeds from all bleed events
RIAL_USD_PRICE = 853_661  # TGJU IRR/USD
PRICE_PER_UI_FACTOR_VIII = 58_000  # IRR, FDA
IR_PROPHYLAXIS_WEEKLY_DOSE = 25 * 2  # IR Protocol
STANDARD_PROPHYLAXIS_WEEKLY_DOSE = 25 * 3  # IR Protocol
BLEEDING_DOSE = 30 * 4  #  Guideline average
JOINT_BLEEDING_DOSE = 30 * 2  #  Guideline average
LT_BLEEDING_DOSE = 550  #  Guideline average
WILLINGNESS_TO_PAY_THRESHOLD = 4_771.4 * 3  # USD

# Will be ignored if None value passthrough
DISCOUNT_RATE_WEEKLY = None
# DISCOUNT_RATE_WEEKLY = ((1 + 0.035) ** (1 / 52)) - 1

PPP_CONVERSION_FACTOR = 117_170  # World Bank 2024 IRR/USD, PPP
# PPP_CONVERSION_FACTOR = 165_354  # Y_CHART | IMF 2025 IRR/USD, PPP

# PLACEHOLDER VALUES
HEMARTHROSIS_TO_ARTHROPATHY = 0.0015  # Probability of joint bleed to develop chronic arthropathy