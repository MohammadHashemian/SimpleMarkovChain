from typing import List, Dict, Literal, Tuple
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from statsmodels.regression.linear_model import OLSResults
from statsmodels.robust.robust_linear_model import RLMResults
from src.utils.logger import get_logger
from matplotlib.lines import Line2D
import model.constants as constants
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import pandas as pd

# TODO:
# Maybe removing outliers before simulating ICER

logger = get_logger()


def remove_outliers(
    df: pd.DataFrame, endog_col: str, exog_col: str, threshold_factor=4
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


# TODO:
# Maybe caching? or passing to plots instead of args
def extract(*args) -> dict[Literal["dataframes", "icer_pairs", "categorized"], List]:
    """
    Summary
    -------
    Instead of unzipping results for every plot, it provides un zipped data, even as pairs, ready to plot.

    Args:
        *args: data passed to plot functions from master plot() function, used to unwrap model inputs and outputs

    Returns:
        dict: type hinted
    """

    # Extracting arguments
    (
        on_demand_inputs,
        prophylaxis_inputs,
        on_demand_results,
        prophylaxis_results,
        n_samples,
    ) = args

    # Extract data for plotting
    # Features | Predictors
    on_demand_abr = [inp["abr"] for inp in on_demand_inputs]
    prophylaxis_abr = [inp["abr"] for inp in prophylaxis_inputs]
    on_demand_ajbr = [inp["ajbr"] for inp in on_demand_inputs]
    prophylaxis_ajbr = [inp["ajbr"] for inp in prophylaxis_inputs]
    # Outcomes | Dependent
    on_demand_utilities = on_demand_results["QALYS"]
    on_demand_consumptions = on_demand_results["total_factors_use"]
    on_demand_annual_use = on_demand_results["annual_factor_consumption"]
    on_demand_costs = on_demand_results["total_factors_costs"]
    prophylaxis_costs = prophylaxis_results["total_factors_costs"]
    prophylaxis_consumptions = prophylaxis_results["total_factors_use"]
    prophylaxis_annual_use = prophylaxis_results["annual_factor_consumption"]
    prophylaxis_utilities = prophylaxis_results["QALYS"]

    # Creating Dataframe for easier handling and masking
    on_demand_df = pd.DataFrame(
        {
            "abr": on_demand_abr,
            "ajbr": on_demand_ajbr,
            "consumption": on_demand_consumptions,
            "annual_factor_use": on_demand_annual_use,
            "costs": on_demand_costs,
            "qalys": on_demand_utilities,
        }
    )
    prophylaxis_df = pd.DataFrame(
        {
            "abr": prophylaxis_abr,
            "ajbr": prophylaxis_ajbr,
            "consumption": prophylaxis_consumptions,
            "annual_factor_use": prophylaxis_annual_use,
            "costs": prophylaxis_costs,
            "qalys": prophylaxis_utilities,
        }
    )
    # Dropping outliers using cook's distance
    on_demand_df = remove_outliers(on_demand_df, "costs", "abr")
    prophylaxis_df = remove_outliers(prophylaxis_df, "costs", "abr")

    # Prepare (Cost, QALY, ABR) pairs
    on_demand_pair = [
        (row["costs"], row["qalys"], row["abr"]) for _, row in on_demand_df.iterrows()
    ]
    prophylaxis_pair = [
        (row["costs"], row["qalys"], row["abr"]) for _, row in prophylaxis_df.iterrows()
    ]

    # Removes Increase in ABR ICER, Impossible transition
    icer_pairs = [
        (p_p[0] - o_p[0], p_p[1] - o_p[1], p_p[2] - o_p[2])
        for o_p in on_demand_pair
        for p_p in prophylaxis_pair
        if not (p_p[2] - o_p[2]) > 0
    ]
    logger.info(
        f"Possible transitions nodes from on_demand to prophylaxis: {len(icer_pairs)}"
    )

    # Categorize ICERs
    dom, dmd, ce, nce, lce, icers = [], [], [], [], [], []

    for dc, dq, da in icer_pairs:
        if da > 0:
            raise ValueError("ICER calculation with positive delta ABR is prohibited")
        if dq < 0 and dc > 0:  # Dominated: lower effectiveness, higher cost
            pair = (float("inf"), (dc, dq, da))  # Use inf for dominated ICERs
            dmd.append(pair)
            continue
        if dq < 0 and dc < 0:  # Lower effectiveness, lower cost
            pair = (float("inf"), (dc, dq, da))  # Use inf for ICERs
            dmd.append(pair)
            continue
        if dq == 0:  # Handle zero QALYs to avoid division by zero
            pair = (float("inf"), (dc, dq, da))
            nce.append(pair)
            continue
        icer = dc / dq
        icers.append(icer)
        pair = (icer, (dc, dq, da))
        if dc < 0 and dq > 0:  # Dominant: cost-saving, better outcome
            dom.append(pair)
        elif icer <= constants.WILLINGNESS_TO_PAY_THRESHOLD:  # Cost-effective
            ce.append(pair)
        else:  # Not cost-effective
            nce.append(pair)
    logger.info("Categorized ICERS pairs:")
    logger.info(f"Dominant: {len(dom)} with portion: {(len(dom)/len(icer_pairs)):.2f}%")
    logger.info(
        f"Dominated: {len(dmd)} with portion: {(len(dmd)/len(icer_pairs)):.2f}%"
    )
    logger.info(
        f"Lower cost with lower effectiveness: {len(lce)}, portion: {(len(lce)/len(icer_pairs)):.4f}%"
    )
    logger.info(
        f"Cost-effective: {len(ce)} with portion: {(len(ce)/len(icer_pairs)):.2f}%"
    )
    logger.info(
        f"Not Cost-effective: {len(nce)} with portion: {(len(nce)/len(icer_pairs)):.2f}%"
    )
    try:
        assert np.isclose(
            len(icer_pairs),
            len(dom) + len(dmd) + len(ce) + len(nce),
        )
        logger.info("Categorization asserted")
    except AssertionError:
        logger.warning("Data lost during categorization")

    output = {
        "dataframes": [on_demand_df, prophylaxis_df],
        "icer_pairs": icer_pairs,
        "categorized": [dom, dmd, ce, nce, icers],
    }
    return output  # type: ignore


# --- Plot 1: Scatter plot (Factor Consumption vs. ABR) ---
def plot_consumption_vs_abr(*args):

    data_dict = extract(*args)
    on_demand_df, prophylaxis_df = data_dict["dataframes"]
    scatter_fig = plt.figure(figsize=(12, 8))
    scatter_ax: Axes = scatter_fig.add_subplot(1, 1, 1)

    # Scatter plot for factor consumption
    scatter1 = scatter_ax.scatter(
        on_demand_df["abr"],
        on_demand_df["consumption"],
        c=on_demand_df["ajbr"],
        cmap="viridis",
        label="On-Demand",
        alpha=0.6,
        s=50,
    )
    scatter2 = scatter_ax.scatter(
        prophylaxis_df["abr"],
        prophylaxis_df["consumption"],
        c=prophylaxis_df["ajbr"],
        cmap="plasma",
        label="Prophylaxis",
        marker="^",
        alpha=0.6,
        s=50,
    )

    # Robust regression for On-Demand (factor consumption)
    X_od = sm.add_constant(on_demand_df["abr"])
    rlm_od = sm.RLM(on_demand_df["consumption"], X_od, M=sm.robust.norms.HuberT())
    od_rlm_results: RLMResults = rlm_od.fit()  # type: ignore
    scatter_ax.plot(
        on_demand_df["abr"],
        od_rlm_results.predict(X_od),
        "b--",
        label=f"On-Demand (robust): slope={od_rlm_results.params.iloc[1]:.2f}",
    )

    # Robust regression for Prophylaxis (factor consumption)
    X_pro = sm.add_constant(prophylaxis_df["abr"])
    rlm_pro = sm.RLM(prophylaxis_df["consumption"], X_pro, M=sm.robust.norms.HuberT())
    prophylaxis_rlm_results: RLMResults = rlm_pro.fit()  # type: ignore
    scatter_ax.plot(
        prophylaxis_df["abr"],
        prophylaxis_rlm_results.predict(X_pro),
        "r--",
        label=f"Prophylaxis (robust): slope={prophylaxis_rlm_results.params.iloc[1]:.2f}",
    )

    scatter_ax.set_xlabel("Annual Bleeding Rate (ABR)")
    scatter_ax.set_ylabel("Total Factor Consumption (Units)")
    scatter_ax.set_title("Factor Consumption vs. ABR")
    scatter_ax.legend()
    scatter_ax.grid(True, alpha=0.3)
    scatter_fig.colorbar(scatter1, ax=scatter_ax, label="AJBR (On-Demand)")

    logger.info("Factor consumption over ABR plotted")
    return scatter_fig


# --- Plot 2: Histogram (Factor Consumption Distribution) ---
def plot_consumption_hist(*args):
    (
        on_demand_inputs,
        prophylaxis_inputs,
        on_demand_results,
        prophylaxis_results,
        n_samples,
    ) = args
    # Extract data for plotting
    on_demand_factors = on_demand_results["total_factors_use"]
    prophylaxis_factors = prophylaxis_results["total_factors_use"]
    hist_fig = plt.figure(figsize=(10, 8))
    hist_ax: Axes = hist_fig.add_subplot(1, 1, 1)
    bins = min(10, max(5, n_samples // 5))  # Adaptive bins
    hist_ax.hist(
        on_demand_factors,
        bins=bins,
        alpha=0.5,
        label="On-Demand",
        color="teal",
    )
    hist_ax.hist(
        prophylaxis_factors,
        bins=bins,
        alpha=0.5,
        label="Prophylaxis",
        color="purple",
    )
    hist_ax.set_xlabel("Total Factor Consumption (Units)")
    hist_ax.set_ylabel("Frequency")
    hist_ax.set_title("Distribution of Factor Consumption")
    hist_ax.legend()
    hist_ax.grid(True, alpha=0.3)
    return hist_fig


# Working on
# --- Plot 3: Scatter plot (Costs vs. ABR) ---
def plot_costs_vs_abr(*args):
    data_dict = extract(*args)

    # load as dataframe for easier handling and masking
    on_demand_df, prophylaxis_df = data_dict["dataframes"]

    # Create figure and axis
    cost_fig = plt.figure(figsize=(12, 8))
    cost_ax: Axes = cost_fig.add_subplot(1, 1, 1)

    # Scatter plot for filtered data
    cost_scatter1 = cost_ax.scatter(
        x=on_demand_df["abr"],
        y=on_demand_df["costs"],
        c=on_demand_df["ajbr"],
        cmap="viridis",
        label="On-Demand (Filtered)",
        alpha=0.6,
        s=50,
    )
    cost_scatter2 = cost_ax.scatter(
        x=prophylaxis_df["abr"],
        y=prophylaxis_df["costs"],
        c=prophylaxis_df["ajbr"],
        cmap="plasma",
        label="Prophylaxis (Filtered)",
        marker="^",
        alpha=0.6,
        s=50,
    )

    # Robust regression for filtered On-Demand
    X_od_filtered = sm.add_constant(on_demand_df["abr"])
    rlm_od_cost = sm.RLM(
        on_demand_df["costs"], X_od_filtered, M=sm.robust.norms.HuberT()
    )
    od_rlm_cost_results: RLMResults = rlm_od_cost.fit()  # type: ignore
    cost_ax.plot(
        on_demand_df["abr"],
        od_rlm_cost_results.predict(X_od_filtered),
        "b--",
        label=f"On-Demand (robust): slope={od_rlm_cost_results.params.iloc[1]:.2f}",
    )

    # Robust regression for filtered Prophylaxis
    X_pro_filtered = sm.add_constant(prophylaxis_df["abr"])
    rlm_pro_cost = sm.RLM(
        prophylaxis_df["costs"], X_pro_filtered, M=sm.robust.norms.HuberT()
    )
    pro_rlm_cost_results: RLMResults = rlm_pro_cost.fit()  # type: ignore
    cost_ax.plot(
        prophylaxis_df["abr"],
        pro_rlm_cost_results.predict(X_pro_filtered),
        "r--",
        label=f"Prophylaxis (robust): slope={pro_rlm_cost_results.params.iloc[1]:.2f}",
    )

    # Axis labels and styling
    cost_ax.set_xlabel("Annual Bleeding Rate (ABR)")
    cost_ax.set_ylabel("Mean Annual Factor Cost (USD)")
    cost_ax.set_title("Factor Costs vs. ABR (Outliers Removed via Cook’s Distance)")
    cost_ax.legend()
    cost_ax.grid(True, alpha=0.3)
    cost_fig.colorbar(cost_scatter1, ax=cost_ax, label="AJBR (On-Demand)")
    return cost_fig


# --- Plot 4: Scatter plot (QALYS vs. ABR) ---
def plot_qaly_vs_abr(*args):
    (
        on_demand_inputs,
        prophylaxis_inputs,
        on_demand_results,
        prophylaxis_results,
        n_samples,
    ) = args

    # Extract data for plotting
    on_demand_abr = [inp["abr"] for inp in on_demand_inputs]
    on_demand_utilities = on_demand_results["QALYS"]
    prophylaxis_abr = [inp["abr"] for inp in prophylaxis_inputs]
    prophylaxis_utilities = prophylaxis_results["QALYS"]

    utility_fig = plt.figure(figsize=(12, 8))
    utility_ax = utility_fig.add_subplot(1, 1, 1)

    # Scatter plot for on_demand group
    scatter1 = utility_ax.scatter(
        on_demand_abr,
        on_demand_utilities,
        label="On-Demand",
        c=on_demand_utilities,
        cmap="viridis_r",
        alpha=0.6,
        s=50,  # Increase marker size for visibility
    )

    # Scatter plot for prophylaxis group
    scatter2 = utility_ax.scatter(
        prophylaxis_abr,
        prophylaxis_utilities,
        label="Prophylaxis",
        c=prophylaxis_utilities,
        cmap="plasma",
        marker="^",
        alpha=0.6,
        s=50,
    )

    plt.colorbar(scatter2, ax=utility_ax, label="Prophylaxis QALYs", pad=0.01)
    plt.colorbar(scatter1, ax=utility_ax, label="On-Demand QALYs", pad=0.02)

    utility_ax.set_xlabel("Annual Bleeding Rate (ABR)")
    utility_ax.set_ylabel("Discounted QALYs")
    utility_ax.set_title("QALYs vs. Annual Bleeding Rate")
    utility_ax.legend()
    utility_ax.grid(True, linestyle="--", alpha=0.7)  # Add grid for readability
    return utility_fig


# --- Plot 5: Scatter plot (Costs vs. QALYs) ---
def plot_icer_scatter(*args):
    icer_fig = plt.figure(figsize=(20, 10))
    icer_ax = icer_fig.add_subplot(1, 1, 1)

    icer_pairs, (dominant, dominated, cost_eff, not_cost_eff, icers) = extract(*args)

    # Summary
    delta_costs, delta_qalys, delta_abr = (
        zip(*icer_pairs) if icer_pairs else ([], [], [])
    )
    logger.info(
        f"Max reduction on weeks spent with bleeding: {np.max(delta_abr) if delta_abr else 'N/A'} weeks"
    )
    logger.info(
        f"Dominant: {len(dominant)}, Cost-effective: {len(cost_eff)}, Not cost-effective: {len(not_cost_eff)}, Dominated: {len(dominated)}"
    )

    def median_icer(pairs):
        values = [i for i, _ in pairs if np.isfinite(i)]
        return np.median(values) if values else float("nan")

    for label, pairs in [
        ("dominant", dominant),
        ("cost-effective", cost_eff),
        ("not cost-effective", not_cost_eff),
    ]:
        med = median_icer(pairs)
        logger.info(
            f"Median ICER ({label}): {'N/A' if np.isnan(med) else f'${med:,.2f}/QALY'}"
        )

    # Centroid
    m_cost, m_qaly = np.median(delta_costs or [0]), np.median(delta_qalys or [0])
    centroid = m_cost / m_qaly if m_qaly else float("inf")
    logger.info(
        f"Median ICER: {'Undefined' if m_qaly == 0 else f'{centroid:,.0f}/QALY'}, Median ΔABR: {np.median(delta_abr):.2f}"
    )

    # Plot scatter
    def scatter(pairs, label, marker, cmap="viridis", color=None):
        if not pairs:
            logger.warning(f"No pair value passed to {label} figure")
            return None
        _, data = zip(*pairs)
        dc, dq, da = zip(*data)
        return icer_ax.scatter(
            dq,
            dc,
            c=da,
            cmap=cmap,
            color=color,
            marker=marker,
            alpha=0.6,
            s=1,
            label=label,
        )

    s_dmd = scatter(
        dominated, "Dominated (More Cost, Worse Outcome)", "v", cmap="Grays"
    )
    s_dom = scatter(dominant, "Dominant (Cost-Saving, More Effective)", "^")
    s_ce = scatter(cost_eff, "Cost-Effective", "o", cmap="viridis")
    s_nce = scatter(not_cost_eff, "Not Cost-Effective", "x", cmap="plasma")

    # Color bars
    plt.colorbar(s_dmd, ax=icer_ax, label="Dominated ΔABR (weeks)", location="left")
    plt.colorbar(
        s_nce,
        ax=icer_ax,
        label="Not Cost-Effective ΔABR (weeks)",
    )
    plt.colorbar(s_dom, ax=icer_ax, label="Dominant ΔABR (weeks)")

    # Centroid
    icer_ax.scatter(
        m_qaly, m_cost, color="black", marker="*", s=50, label="Centroid ICER", zorder=5
    )
    icer_ax.annotate(
        f"${centroid:,.0f}/QALY",
        (float(m_qaly), float(m_cost)),
        textcoords="offset points",
        xytext=(25, -75),
        ha="center",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="white"),
    )

    # Axes, lines
    x_rng = np.array([min(delta_qalys or [0]) - 0.1, max(delta_qalys or [0]) + 0.1])
    icer_ax.plot(
        x_rng,
        x_rng * constants.WILLINGNESS_TO_PAY_THRESHOLD,
        "k--",
        alpha=0.8,
        label=f"WTP: ${constants.WILLINGNESS_TO_PAY_THRESHOLD:,}/QALY",
    )
    icer_ax.axhline(0, color="gray", linestyle="-", alpha=0.5)
    icer_ax.axvline(0, color="gray", linestyle="-", alpha=0.5)
    icer_ax.set(
        xlabel="Δ QALYs",
        ylabel="Δ Cost ($)",
        title="Cost-Effectiveness Plane",
        ylim=(-20000, 40000),
    )
    icer_ax.grid(True, linestyle="--", alpha=0.7)

    # Custom legend to handle different marker sizes
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            label="Dominated (More Cost, Worse Outcome)",
            markerfacecolor="gray",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label="Dominant (Cost-Saving, More Effective)",
            markerfacecolor="blue",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Cost-Effective",
            markerfacecolor="green",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="w",
            label="Not Cost-Effective",
            markerfacecolor="red",
            markeredgecolor="red",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            label="Centroid ICER",
            markerfacecolor="black",
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            linestyle="--",
            color="black",
            label=f"WTP: ${constants.WILLINGNESS_TO_PAY_THRESHOLD:,}/QALY",
        ),  # WTP line
    ]

    icer_ax.legend(handles=legend_elements, loc="lower right")
    return icer_fig


def plot_icer_histogram(*args):
    pass


# Plot all results
def plot(
    on_demand_inputs: List[Dict],
    prophylaxis_inputs: List[Dict],
    on_demand_results: Dict,
    prophylaxis_results: Dict,
    n_samples: int,
) -> dict[str, Figure]:
    figures = {}
    # Key for plot name
    plot_functions = {
        "factor_consumptions": plot_consumption_vs_abr,
        "factor_histogram": plot_consumption_hist,
        "costs_per_abr": plot_costs_vs_abr,
        "qalys_per_abr": plot_qaly_vs_abr,
        # "incremental_cost_effectiveness": plot_icer_scatter,
        # "icer_histogram": plot_icer_histogram,
    }
    for key, func in plot_functions.items():
        fig = func(
            on_demand_inputs,
            prophylaxis_inputs,
            on_demand_results,
            prophylaxis_results,
            n_samples,
        )
        if not fig:
            logger.warning(f"Function: {func.__name__} returns None type")
            continue
        figures[key] = fig
    return figures
