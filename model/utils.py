from pathlib import Path
from statsmodels.regression.linear_model import OLSResults
from statsmodels.robust.robust_linear_model import RLMResults
from scipy.stats import poisson
from sklearn.preprocessing import normalize
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prob_at_least_one(lam: float) -> float:
    """
    Calculate the probability of at least one event occurring in a given interval.

    Args:
        lam: Mean number of events occurring within the given interval

    Returns:
        Probability of at least one event.
    """
    # Converse probability
    # P(at least one) = 1 - p(failure)**n
    # n: number of trials
    return 1 - np.exp(-lam)


def poisson_mass_function(lam: float, k: int, loc=0):
    """
    Poisson mass function(given k): exp(-λ) * ((λ)**k)/ k!
    Args:
        lam: λ
        k: number of expected events
        loc: to shift distribution, 0 to standardized form by default
    """
    return poisson.pmf(k=k, mu=lam, loc=loc)


def zero_truncated_mass_function(
    lam: int | float | np.number | np.int64, k: int | float | np.number
) -> float:
    """
    Zero-Truncated Poisson PMF: (λ**k) / ((e**λ) -1) * k!
    Args:
        k: value(s) to evaluate the PMF at (must be integer >= 1).
        lam: rate parameter of the underlying poisson distribution.
    """
    # The classic ZTP formula
    if not isinstance(k, int | float | np.number):
        raise TypeError(f"Invalid input, expected number value, got {type(k)}")
    if k == 0:
        raise ValueError("zero is truncated")
    numerator = np.power(lam, k)
    denominator = (math.exp(lam) - 1) * math.factorial(int(k))
    res = numerator / denominator
    return res


def cal_body_weight(week: int, b: int = 0) -> float:
    """
    Estimates male body weight in kg using Gompertz growth model (0-50 years)
    and linear decline (50-73 years) based on WHO/CDC data.
    Does not support numpy vectorization yet

    Gompertz parameters optimized for key milestones:
    - Birth (0 weeks): 3.3 kg
    - 1 year (52 weeks): 10.0 kg
    - 18 years (936 weeks): 70.0 kg
    - 50 years (2600 weeks): 80.0 kg

    Args:
        week (int): Age in weeks (0 to 3796)
        b (int): f(x0) = b

    Returns:
        float: Estimated weight in kg, rounded to 2 decimals

    Raises:
        ValueError: For invalid input
    """
    week += b
    if not isinstance(week, int) or week < 0 or week > 3796:
        raise ValueError("Week must be an integer between 0 and 3796")

    # Optimized Gompertz parameters for growth phase (0-2600 weeks)
    A = 80.5  # Asymptotic weight (kg)
    B = 3.08  # Displacement parameter
    K = 0.00255  # Growth rate

    if week <= 2600:
        # Gompertz growth model
        weight = A * math.exp(-B * math.exp(-K * week))
    else:
        # Linear decline from 50-73 years (80kg@2600wks → 75kg@3796wks)
        weight = 80.0 - (5.0 * (week - 2600) / (3796 - 2600))
    return round(weight, 2)


def plot_body_weight():
    # Generate denser points
    weeks = np.arange(0, 3640, 10)  # Every 10 weeks
    weights = [cal_body_weight(int(w), b=2 * 52) for w in weeks]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(weeks, weights, "b-", label="Male Body Weight")

    plt.xlim(0, 3796)
    plt.xlabel("Age (weeks)")
    plt.ylabel("Weight (kg)")
    plt.title("Male Body Weight Growth (Birth to 73 Years)")
    plt.grid(True, which="both", ls="--")
    plt.legend()

    # Add key age markers
    key_ages = [0, 52, 520, 936, 2600, 3796]
    key_labels = ["", "1 yr", "10 yrs", "18 yrs", "50 yrs", "73 yrs"]
    plt.xticks(key_ages, key_labels)

    # Add annotations
    for w, label in zip(key_ages, key_labels):
        weight = cal_body_weight(w)
        plt.text(w, weight, f"{weight} kg", fontsize=10, ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs" / "figures" / "body_weight.png")


def remove_outliers(
    df: pd.DataFrame, endog_col: str, exog_col: str, threshold_factor: float = 4
) -> pd.DataFrame:
    """
    Helper function to remove outliers, supports pandas dataframe
    """
    X_constant = sm.add_constant(df[exog_col])
    ols: OLSResults = sm.OLS(endog=df[endog_col], exog=X_constant).fit()  # type: ignore
    cooks_d = ols.get_influence().cooks_distance[0]
    threshold = threshold_factor / len(df)
    mask = cooks_d <= threshold

    filtered_df = df[mask]
    # Print number of outliers removed
    print(f"Removing {len(df) - len(filtered_df)} outliers")
    return filtered_df


def remove_outliers_robust(
    df: pd.DataFrame, endog_col: str, exog_col: str, threshold_factor: float = 4
) -> pd.DataFrame:
    """
    Helper function to remove outliers using robust linear regression, supports pandas dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing the endogenous and exogenous variables.
    endog_col : str
        Name of the column containing the dependent variable.
    exog_col : str
        Name of the column containing the independent variable.
    threshold_factor : float, optional
        Factor to determine the threshold for outlier detection (default is 4).

    Returns:
    --------
    pd.DataFrame
        Dataframe with outliers removed based on approximate Cook's distance.
    """
    # Add constant term for intercept
    X_constant = sm.add_constant(df[exog_col])

    # Fit robust linear model using RLM with M-estimator (HuberT is default)
    rlm: RLMResults = sm.RLM(
        endog=df[endog_col], exog=X_constant, M=sm.robust.norms.HuberT()
    ).fit()  # type: ignore

    # Calculate approximate Cook's distance for robust regression
    # Get standardized residuals
    resid = rlm.resid / rlm.scale

    # Calculate leverage (hat matrix diagonal)
    hat_matrix_diag = np.diag(
        rlm.model.exog
        @ np.linalg.pinv(rlm.model.exog.T @ rlm.model.exog)
        @ rlm.model.exog.T
    )

    # Approximate Cook's distance: (standardized residual)^2 * leverage / (p * (1 - leverage))
    p = X_constant.shape[1]  # Number of parameters
    cooks_d_approx = (resid**2 * hat_matrix_diag) / (
        p * (1 - hat_matrix_diag + 1e-10)
    )  # Add small constant to avoid division by zero

    # Set threshold for outlier detection
    threshold = threshold_factor / len(df)
    mask = cooks_d_approx <= threshold

    # Filter dataframe to remove outliers
    filtered_df = df[mask]

    # Print number of outliers removed
    print(f"Removing {len(df) - len(filtered_df)} outliers")

    return filtered_df


# DEPRECATED
def normalize_to_sum_to_one(array: list[float] | np.ndarray) -> np.ndarray:
    """
    It's not actually fast, actually not really useful in my case
    """
    if isinstance(array, list):
        array = np.array(array)
    if len(array.shape) > 1:
        raise ValueError("Array have more than 1-Dimension")
    if np.any(array < 0):
        raise ValueError("Array contains negative values")
    normalized = normalize(array.reshape(-1, 1), norm="l1", axis=0).ravel()
    return normalized


# DEPRECATED
def count_bleeds_conditional_prob(state: str, **kwargs) -> int:
    # Conditional probability formula
    def conditional_probs(k: int, lam: float):
        return (lam**k) / (math.factorial(k) * (math.exp(lam) - 1))

    # Get lambda_value from kwargs
    lambda_value = (
        kwargs.get("lambda_bleeding")
        if state.lower() == "bleeding"
        else (
            kwargs.get("lambda_joint_bleeding")
            if state.lower() == "joint_bleeding"
            else 0
        )
    )
    number_of_bleeds = 1
    if lambda_value != 0:
        if not isinstance(lambda_value, float):
            raise TypeError("No valid lambda value provided.")
        events_probs = [conditional_probs(k, lambda_value) for k in range(1, 5, 1)]
        # Normalizing
        events_probs = [p / sum(events_probs) for p in events_probs]
        number_of_bleeds = np.random.choice([i for i in range(1, 5, 1)], p=events_probs)
    return number_of_bleeds
