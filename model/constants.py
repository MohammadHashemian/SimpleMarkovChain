from model.defined_types import Currencies


DEVELOPMENT = True
BASE_SOBOL_SAMPLE_SIZE = 512  # Final sample size follows: N * (D + 2)

# Overrides to smaller size to reduce computation
if DEVELOPMENT:
    BASE_SOBOL_SAMPLE_SIZE = 64

# [------- UTILITIES -------]
STATE_UTILITIES = {
    "Healthy": 0.915,
    "Mild_Arthropathy": 0.85,
    "Moderate_Arthropathy": 0.80,
    "Severe_Arthropathy": 0.75,
    # New model structure uses dynamic disease progression using pettersson categories
    "Arthropathy": 0.75,  # Placeholder, DEPRECATED
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
            else "Moderate_Arthropathy" if i <= 28 else "Severe_Arthropathy"
        )
    )
    for i in range(80)
}
# Annually over 1000 population 7.76 person dies 2024 records
MORTALITY_RATE = 7.76 / 1000

# # [------- ECONOMICS -------]
MODEL_CURRENCY = Currencies.IRR
# Dollar
GDP_PER_CAPITA = 4_771.4  # USD
WTP_THRESHOLD = GDP_PER_CAPITA * 3  # USD
REPORT_PPP = False  # Reporting PPP works with USD currency
PPP_CONVERSION_FACTOR = 117_170  # World Bank 2024 IRR/USD, PPP
# PPP_CONVERSION_FACTOR = 165_354  # Y_CHART | IMF 2025 IRR/USD, PPP

# IGNORED FEATURES (expenditures and effects are at same time, not needing to discount in this case)
# DISCOUNT_RATE_WEEKLY = ((1 + 0.035) ** (1 / 52)) - 1
DISCOUNT_RATE_WEEKLY = None

if MODEL_CURRENCY.value.upper() == "IRR":
    # Toman (IRR)
    GDP_PER_CAPITA = 180_000_000
    # For orphan disease
    WTP_THRESHOLD = GDP_PER_CAPITA * 10  # 180 MIllion toman

elif MODEL_CURRENCY.value.upper() == "TOMAN":
    GDP_PER_CAPITA = 180_000_000 / 10
    # For orphan disease
    WTP_THRESHOLD = GDP_PER_CAPITA * 10  # 180 MIllion toman

# [------- SIMULATION STEPS -------]
WOY = 52  # Number of weeks per year
# Primary prophylaxis simulates for 10 years of continuos factor therapy starting from age 2 till 12 (pediatrics/children)
# Lifetime horizon simulation includes patients from 2 years of age till 100 (children/pediatrics/adolescents/adults)
# Secondary prophylaxis considered whom received prophylactic therapy later in life, 88 years in weeks (12, 100) (adolescents/adults)
PRIMARY_CYCLE_COUNT = 10 * WOY  # 520
LIFETIME_CYCLE_COUNTS = 98 * WOY  # 5096
SECONDARY_CYCLE_COUNTS = 88 * WOY  # 4576

# Pediatrics starts therapy from age 2
# Adolescents (delayed) prophylaxis from age of 12, then -> (2 + 10) * 52 = 624
PEDIATRIC_STARTING_POINT = 2 * WOY
ADOLESCENT_STARTING_POINT = PEDIATRIC_STARTING_POINT + PRIMARY_CYCLE_COUNT

# [------- MODEL STRUCTURE -------]
START_STATE = "Healthy"
STATES = [
    "Healthy",
    "Bleeding",
    "Hemarthrosis",
    "LT_Bleeding",
    "Death",
]

ON_DEMAND_ABR_REPORTS = [
    [58.3, 26.9],  # Zhao et al. (median 12 y 1-50 y) 10.1177/1076029621989811
    [37.2, 19.9],  # Manco-Johnson MJ et al. (12-50) 10.1111/jth.13811
    [19.5, 15.0],  # Tagliaferri A et al. (12-25 y) 10.1160/TH14-05-0407
    [17.7, 11.7],  # Tagliaferri A et al. (26-55 y) 10.1160/TH14-05-0407
    # [12.96, 0],  # Gringeri A et al. (1-7 y median 4 y) 10.1111/j.1538-7836.2011.04214.x
    # [13, 12],  # Gringeri A et al. synthetic to flatten the estimation chart
    [7.4, 9.5],  # Romanová G et al. (>=18 y, n=302) 10.1007/s00277-023-05453-6
    # Berntorp E et al. (0-60 Children heavy) 10.1111/hae.13111
    # [13.2, 12.43],  # Berntorp E et al. (0-60 Children heavy)  Deprecated
    [16.8, 10.0],  # Belgium
    [13.8, 12.6],  # France
    [12.2, 18.1],  # Germany
    [11.5, 11.5],  # Italy
    [7.0, 6.4],  # Spain
    [4.5, 0.7],  # Sweden
    [19.4, 10.6],  # UK
    # Khair K et al. (median 17 y - n:299) 10.1111/hae.13361
    [14.0, 12.3],  # First year
    [18.4, 14.3],  # Second year
    [15.8, 8.13],  # Third year
    [58.9, 16.6],  # Zhao et al. (2-12 y, n=30) 10.1080/08880018.2017.1313921
    [5.6, 1.83],  # Eshghi et al. (<15 y, n=24) 10.1177/1076029616685429
    [13.9, 4.47],  # Roberto Musso (23.6 y, n=220) 10.1160/TH07-06-0409
    # [7.0, 7.9],  # R. Klamroth (26y; 86.6% severe/mod-severe) # 10.111/hae.12941
    [57.7, 24.6],  # K. Kavakli (12-65 mean 28y) 10.1111/jth.12828
    # Fukutake (352 PTPs, 75.6% severe, 1-76 mean 25.8 y) # 10.1007/s12185-018-02574-x
    [37.9, 33.08],
    [22.2, 7.0],  # Ying Liu ,(n=34; 4-18y (mean 12.2y) # 10.1111/hae.14016
    [17.7, 9.3],  # B Warren, n:37, 2.5 up to 7.5y, 10.1182/bloodadvances.2019001311
    [17.69, 9.25],  # Marilyn J. Manco-Johnson, <1.5y to 6y, n:65 10.1056/NEJMoa067659
    [13, 9],  # Melissa Kern, n:15, pre target joint 10.1016/j.jpeds.2004.06.082
    [24, 12],  # Melissa Kern, n:15, post target joint 10.1016/j.jpeds.2004.06.082
    # A. Tagliaferri, n: 83, median 23.6 10-72y, 10.1111/j.1365-2516.2008.01791.x
    [35.8, 24.8],
    [21.42, 9.59],  # Aznar, n: 15, 26-47 mean 35.6, 10.1111/vox.12066
    [35.7, 22.2],  # von Drygalski, 26 adults (mean 42.8 y); 10.1056/NEJMoa2209226
    [22.2, 7.0],  # Liu, ODT group (n=18) age 12.4 SD 4.1, 10.1111/hae.14016
]
PROPHYLAXIS_ABR_REPORTS = [
    [2.5, 4.6],  # Zhao et al. (median 12 y 1-50 y) 10.1177/1076029621989811
    [2.5, 4.7],  # Manco-Johnson MJ et al. (12-50) 10.1111/jth.13811
    [2.6, 2.2],  # Tagliaferri A et al. (12-25 y) 10.1160/TH14-05-0407
    [4.5, 7.1],  # Tagliaferri A et al. (12-25 y) 10.1160/TH14-05-0407
    # [6.24, 0],  # Gringeri A et al. (1-7 y median 4 y) 10.1111/j.1538-7836.2011.04214.x
    # [6.3, 6],  # Gringeri A et al. synthetic to flatten the estimation chart
    [2.1, 2.1],  # Romanová G et al. (>=18 y, n=302) 10.1007/s00277-023-05453-6
    # Berntorp E et al. (0-60 Children heavy) 10.1111/hae.13111
    # [4.26, 5.97],  # Berntorp E et al. (0-60 Children heavy) Deprecated
    [4.1, 6.9],  # Belgium
    [8.0, 9.4],  # France
    [4.5, 5.3],  # Germany
    [2.8, 4.7],  # Italy
    [2.7, 2.5],  # Spain
    [1.9, 2.9],  # Sweden
    [5.8, 7.0],  # UK
    # Khair K et al. (median 17 y - n:299) 10.1111/hae.13361
    [3.5, 4.3],  # First year
    [3.3, 4.1],  # Second year
    [3.7, 3.9],  # Third year
    [3.0, 5.9],  # Zhao et al. (2-12 y, n=30) 10.1080/08880018.2017.1313921
    [1.86, 1.52],  # Eshghi et al. (<15 y, n=24) 10.1177/1076029616685429
    [4.8, 5.0],  # Roberto Musso (23.6 y, n=220) 10.1160/TH07-06-0409
    # [5.0, 5.8],  # R. Klamroth (26y; 86.6% severe/mod-severe) # 10.1111/hae.12941
    [4.3, 6.5],  # K. Kavakli (12-65 mean 28y) 10.1111/jth.12828
    # Fukutake (352 PTPs, 75.6% severe, 1-76 mean 25.8 y) # 10.1007/s12185-018-02574-x
    [8.9, 19.61],
    #  Beth Boulden Warren, n:37, 2.5 up to 18y, 10.1182/bloodadvances.2019001311
    [3.5, 2.1],  # Early proph group
    [6.2, 5.3],  # Post-proph group
    [3.27, 6.24],  # Marilyn J. Manco-Johnson, <1.5y to 6y, n:65 10.1056/NEJMoa067659
    # A. Tagliaferri, n: 83, median 23.6 10-72y, 10.1111/j.1365-2516.2008.01791.x
    [4.2, 3.7],
    [1.82, 2.87],  # Aznar, n: 15, 26-47 mean 35.6, 10.1111/vox.12066
    [3.2, 5.4],  # von Drygalski, 133, ≥12 y (mean 33.9 y); 10.1056/NEJMoa2209226
]
DECREMENT_PER_BLEED = {
    "on_demand": 0.0003725,
    "prophylaxis": 0.0018,
}  # Placeholder values
AJBR_FRACTION = 0.75  # Percentage of joint bleeds from all bleed events
LTB_FRACTION = 0.045  # Percent of life threatening bleeds from all bleed events
OD_LTB_RATE = 255 / 100_000  # TODO
PRO_LTB_RATE = 166 / 100_000  # TODO
RIAL_USD_PRICE = 853_661  # TGJU IRR/USD
PRICE_PER_UI_FACTOR_VIII = 58_000  # IRR, FDA
IR_PROPHYLAXIS_WEEKLY_DOSE = 25 * 2  # IR Protocol
STANDARD_PROPHYLAXIS_WEEKLY_DOSE = 25 * 3  # IR Protocol
PETTERSSON_CONVERSION_FACTOR = 12.6

# 87.3% of bleeding episodes resolved with one injection #NCT01181128. (Blood. 2014;123(3):317-325)
# Guideline averages or PSA (?)
BLEEDING_DOSE = 30 * 4
JOINT_BLEEDING_DOSE = 30 * 2
LT_BLEEDING_DOSE = 550

# TODO:
WILLINGNESS_TO_PAY_THRESHOLD_IRR = None
LATE_ARTHROPATHY = None

# Rate indicates frequency of Hemarthrosis causes permanent joint damage (lambda)
# Tuned to fit Manco-Johnson M et al. article on NEW ENGLAND journal:
# Radio graphic results indicate 93% and 81% had no joint damage
# MRI suggests 93% of prophylaxis and 55% of on demand had normal joint
EARLY_ARTHROPATHY = (
    0.05  # (PSA (?) as it's really effects the const-effectiveness results)
)
