from typing import List, Dict, Tuple
from dataclasses import dataclass
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from statsmodels.robust.robust_linear_model import RLMResults
from model.utils import remove_outliers_robust
from src.utils.logger import get_logger
import model.constants as constants
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import pandas as pd

logger = get_logger()


@dataclass
class DataExtract:
    dataframes: Tuple[pd.DataFrame, pd.DataFrame]  # on_demand_df, prophylaxis_df
    icer_pairs: List[Tuple[float, float, float]]  # (delta_cost, delta_qaly, delta_abr)
    categorized: Tuple[
        List[Tuple[float, Tuple[float, float, float]]],  # dominant
        List[Tuple[float, Tuple[float, float, float]]],  # dominated
        List[Tuple[float, Tuple[float, float, float]]],  # cost-effective
        List[Tuple[float, Tuple[float, float, float]]],  # not cost-effective
        List[float],  # icers
    ]


def extract(
    on_demand_inputs: List[Dict],
    prophylaxis_inputs: List[Dict],
    on_demand_results: Dict,
    prophylaxis_results: Dict,
    n_samples: int,
) -> DataExtract:
    """
    Summary
    -------
    Extracts and processes data for plotting, returning a typed DataExtract object.

    Args:
        on_demand_inputs: List of dictionaries with model inputs for on-demand treatment
        prophylaxis_inputs: List of dictionaries with model inputs for prophylaxis treatment
        on_demand_results: Dictionary with model outputs for on-demand treatment
        prophylaxis_results: Dictionary with model outputs for prophylaxis treatment
        n_samples: Number of samples

    Returns:
        DataExtract: Object containing dataframes, ICER pairs, and categorized ICERs
    """
    # Extract data for plotting
    on_demand_abr = [inp["abr"] for inp in on_demand_inputs]
    prophylaxis_abr = [inp["abr"] for inp in prophylaxis_inputs]
    on_demand_ajbr = [inp["ajbr"] for inp in on_demand_inputs]
    prophylaxis_ajbr = [inp["ajbr"] for inp in prophylaxis_inputs]
    on_demand_utilities = on_demand_results["QALYS"]
    on_demand_consumptions = on_demand_results["total_factors_use"]
    on_demand_annual_use = on_demand_results["annual_factor_consumption"]
    on_demand_costs = on_demand_results["total_factors_costs"]
    prophylaxis_costs = prophylaxis_results["total_factors_costs"]
    prophylaxis_consumptions = prophylaxis_results["total_factors_use"]
    prophylaxis_annual_use = prophylaxis_results["annual_factor_consumption"]
    prophylaxis_utilities = prophylaxis_results["QALYS"]

    # Create DataFrames
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
    # Remove outliers
    on_demand_df = remove_outliers_robust(
        on_demand_df, "costs", "abr", threshold_factor=4
    )
    prophylaxis_df = remove_outliers_robust(
        prophylaxis_df, "costs", "abr", threshold_factor=4
    )

    # Prepare (Cost, QALY, ABR) pairs
    on_demand_pair = [
        (row["costs"], row["qalys"], row["abr"]) for _, row in on_demand_df.iterrows()
    ]
    prophylaxis_pair = [
        (row["costs"], row["qalys"], row["abr"]) for _, row in prophylaxis_df.iterrows()
    ]

    icer_pairs = [
        (
            p[0] - o[0],  # Δ Cost
            p[1] - o[1],  # Δ QALY
            p[2] - o[2],  # Δ ABR
        )
        for o, p in zip(on_demand_pair, prophylaxis_pair)
        if (p[2] - o[2]) < 0
    ]

    logger.info(
        f"Possible transitions from on-demand to prophylaxis truncated to: {len(icer_pairs)} pairs"
    )

    # Categorize ICERs
    dom, dmd, ce, nce, lce, icers = [], [], [], [], [], []

    for dc, dq, da in icer_pairs:
        if da > 0:
            raise ValueError("ICER calculation with positive delta ABR is prohibited")
        if dq < 0 and dc > 0:  # Dominated
            pair = (float("inf"), (dc, dq, da))
            dmd.append(pair)
            continue
        if dq < 0 and dc < 0:  # Lower cost, lower effectiveness
            pair = (float("inf"), (dc, dq, da))
            lce.append(pair)
            continue
        if dq == 0:  # Avoid division by zero
            pair = (float("inf"), (dc, dq, da))
            nce.append(pair)
            continue
        icer = dc / dq
        icers.append(icer)
        pair = (icer, (dc, dq, da))
        if dc < 0 and dq > 0:  # Dominant
            dom.append(pair)
        elif icer <= constants.WILLINGNESS_TO_PAY_THRESHOLD:  # Cost-effective
            ce.append(pair)
        else:  # Not cost-effective
            nce.append(pair)

    # Log categorization stats
    total = len(icer_pairs)
    if total > 0:
        logger.info("Categorized ICER pairs:")
        logger.info(f"Dominant: {len(dom)} ({len(dom)/total:.2%})")
        logger.info(f"Dominated: {len(dmd)} ({len(dmd)/total:.2%})")
        logger.info(
            f"Lower cost, lower effectiveness: {len(lce)} ({len(lce)/total:.2%})"
        )
        logger.info(f"Cost-effective: {len(ce)} ({len(ce)/total:.2%})")
        logger.info(f"Not cost-effective: {len(nce)} ({len(nce)/total:.2%})")
    else:
        logger.warning("No ICER pairs to categorize")

    # Verify categorization
    if total != len(dom) + len(dmd) + len(ce) + len(nce) + len(lce):
        logger.warning("Data lost during categorization")

    return DataExtract(
        dataframes=(on_demand_df, prophylaxis_df),
        icer_pairs=icer_pairs,
        categorized=(dom, dmd, ce, nce, icers),
    )


def plot_consumption_vs_abr(data: DataExtract) -> Figure:
    """
    Plot scatter of factor consumption vs. ABR with robust regression lines.
    """
    on_demand_df, prophylaxis_df = data.dataframes
    scatter_fig = plt.figure(figsize=(12, 8))
    scatter_ax: Axes = scatter_fig.add_subplot(1, 1, 1)

    # Scatter plots
    on_demand_scatter = scatter_ax.scatter(
        on_demand_df["abr"],
        on_demand_df["consumption"],
        c=on_demand_df["ajbr"],
        cmap="viridis",
        label="On-Demand",
        alpha=0.4,
        s=25,
    )
    # Prophylaxis scatter
    scatter_ax.scatter(
        prophylaxis_df["abr"],
        prophylaxis_df["consumption"],
        c=prophylaxis_df["ajbr"],
        cmap="plasma",
        label="Prophylaxis",
        marker="^",
        alpha=0.4,
        s=25,
    )

    # Robust regression for On-Demand
    X_od = sm.add_constant(on_demand_df["abr"])
    rlm_od = sm.RLM(on_demand_df["consumption"], X_od, M=sm.robust.norms.HuberT())
    od_rlm_results: RLMResults = rlm_od.fit()  # type: ignore
    scatter_ax.plot(
        on_demand_df["abr"],
        od_rlm_results.predict(X_od),
        "b--",
        label=f"On-Demand (robust): slope={od_rlm_results.params.iloc[1]:.2f}",
    )

    # Robust regression for Prophylaxis
    X_pro = sm.add_constant(prophylaxis_df["abr"])
    rlm_pro = sm.RLM(prophylaxis_df["consumption"], X_pro, M=sm.robust.norms.HuberT())
    pro_rlm_results: RLMResults = rlm_pro.fit()  # type: ignore
    scatter_ax.plot(
        prophylaxis_df["abr"],
        pro_rlm_results.predict(X_pro),
        "r--",
        label=f"Prophylaxis (robust): slope={pro_rlm_results.params.iloc[1]:.2f}",
    )

    scatter_ax.set_xlabel("Annual Bleeding Rate (ABR)")
    scatter_ax.set_ylabel("Total Factor Consumption (Units)")
    scatter_ax.set_title("Factor Consumption vs. ABR")
    scatter_ax.legend()
    scatter_ax.grid(True, alpha=0.3)
    scatter_fig.colorbar(on_demand_scatter, ax=scatter_ax, label="AJBR (On-Demand)")
    logger.info("Factor consumption vs. ABR plotted")
    return scatter_fig


def plot_consumption_hist(data: DataExtract) -> Figure:
    """
    Plot histogram of factor consumption distribution.
    """
    on_demand_df, prophylaxis_df = data.dataframes
    hist_fig = plt.figure(figsize=(10, 8))
    hist_ax: Axes = hist_fig.add_subplot(1, 1, 1)

    # Use number of samples from dataframes
    n_samples = min(len(on_demand_df), len(prophylaxis_df))
    bins = min(10, max(5, n_samples // 5))  # Adaptive bins

    hist_ax.hist(
        on_demand_df["consumption"],
        bins=bins,
        alpha=0.5,
        label="On-Demand",
        color="teal",
    )
    hist_ax.hist(
        prophylaxis_df["consumption"],
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
    logger.info("Factor consumption histogram plotted")
    return hist_fig


def plot_costs_vs_abr(data: DataExtract) -> Figure:
    """
    Plot scatter of costs vs. ABR with robust regression lines.
    """
    on_demand_df, prophylaxis_df = data.dataframes
    cost_fig = plt.figure(figsize=(12, 8))
    cost_ax: Axes = cost_fig.add_subplot(1, 1, 1)

    # Scatter plots
    on_demand_scatter = cost_ax.scatter(
        on_demand_df["abr"],
        on_demand_df["costs"],
        c=on_demand_df["ajbr"],
        cmap="viridis",
        label="On-Demand (Filtered)",
        alpha=0.4,
        s=25,
    )
    # Prophylaxis scatter
    cost_ax.scatter(
        prophylaxis_df["abr"],
        prophylaxis_df["costs"],
        c=prophylaxis_df["ajbr"],
        cmap="plasma",
        label="Prophylaxis (Filtered)",
        marker="^",
        alpha=0.4,
        s=25,
    )

    # Robust regression for On-Demand
    X_od = sm.add_constant(on_demand_df["abr"])
    rlm_od = sm.RLM(on_demand_df["costs"], X_od, M=sm.robust.norms.HuberT())
    od_rlm_results: RLMResults = rlm_od.fit()  # type: ignore
    cost_ax.plot(
        on_demand_df["abr"],
        od_rlm_results.predict(X_od),
        "b--",
        label=f"On-Demand (robust): slope={od_rlm_results.params.iloc[1]:.2f}",
    )

    # Robust regression for Prophylaxis
    X_pro = sm.add_constant(prophylaxis_df["abr"])
    rlm_pro = sm.RLM(prophylaxis_df["costs"], X_pro, M=sm.robust.norms.HuberT())
    pro_rlm_results: RLMResults = rlm_pro.fit()  # type: ignore
    cost_ax.plot(
        prophylaxis_df["abr"],
        pro_rlm_results.predict(X_pro),
        "r--",
        label=f"Prophylaxis (robust): slope={pro_rlm_results.params.iloc[1]:.2f}",
    )

    cost_ax.set_xlabel("Annual Bleeding Rate (ABR)")
    cost_ax.set_ylabel("Mean Annual Factor Cost (USD)")
    cost_ax.set_title("Factor Costs vs. ABR (Outliers Removed via Cook’s Distance)")
    cost_ax.legend()
    cost_ax.grid(True, alpha=0.3)
    cost_fig.colorbar(on_demand_scatter, ax=cost_ax, label="AJBR (On-Demand)")
    logger.info("Costs vs. ABR plotted")
    return cost_fig


def plot_qaly_vs_abr(data: DataExtract) -> Figure:
    """
    Plot scatter of QALYs vs. ABR.
    """
    on_demand_df, prophylaxis_df = data.dataframes
    utility_fig = plt.figure(figsize=(12, 8))
    utility_ax: Axes = utility_fig.add_subplot(1, 1, 1)

    # Scatter plots
    scatter1 = utility_ax.scatter(
        on_demand_df["abr"],
        on_demand_df["qalys"],
        label="On-Demand",
        c=on_demand_df["qalys"],
        cmap="viridis_r",
        alpha=0.4,
        s=25,
    )
    scatter2 = utility_ax.scatter(
        prophylaxis_df["abr"],
        prophylaxis_df["qalys"],
        label="Prophylaxis",
        c=prophylaxis_df["qalys"],
        cmap="plasma",
        marker="^",
        alpha=0.6,
        s=25,
    )

    utility_ax.set_xlabel("Annual Bleeding Rate (ABR)")
    utility_ax.set_ylabel("Discounted QALYs")
    utility_ax.set_title("QALYs vs. Annual Bleeding Rate")
    utility_ax.legend()
    utility_ax.grid(True, linestyle="--", alpha=0.7)
    utility_fig.colorbar(scatter1, ax=utility_ax, label="On-Demand QALYs", pad=0.02)
    utility_fig.colorbar(scatter2, ax=utility_ax, label="Prophylaxis QALYs", pad=0.01)
    logger.info("QALYs vs. ABR plotted")
    return utility_fig


def plot_icer_scatter(data: DataExtract) -> Figure:
    """
    Plot cost-effectiveness plane with ICER scatter.
    """
    icer_fig = plt.figure(figsize=(20, 10))
    icer_ax: Axes = icer_fig.add_subplot(1, 1, 1)
    icer_pairs = data.icer_pairs
    dominant, dominated, cost_eff, not_cost_eff, icers = data.categorized

    # Summary stats
    delta_costs, delta_qalys, delta_abr = (
        zip(*icer_pairs) if icer_pairs else ([], [], [])
    )
    logger.info(
        f"Max reduction in weeks spent with bleeding: {np.max([abs(da) for da in delta_abr]) if delta_abr else 'N/A'} weeks"
    )
    logger.info(
        f"Dominant: {len(dominant)}, Cost-effective: {len(cost_eff)}, Not cost-effective: {len(not_cost_eff)}, Dominated: {len(dominated)}"
    )

    def min_max_median_icer(pairs: List[Tuple[float, Tuple[float, float, float]]]):
        # Pairs icer, (dc, dq, da)
        values = [i for i, _ in pairs if np.isfinite(i)]
        return (np.min(values), np.max(values), np.median(values))

    def min_max_median_abr(
        pairs: List[Tuple[float, Tuple[float, float, float]]],
    ) -> tuple:
        values = [i[2] for _, i in pairs if np.isfinite(_)]
        return (np.min(values), np.max(values), np.median(values))

    for label, pairs in [
        ("dominant", dominant),
        ("cost-effective", cost_eff),
        ("not cost-effective", not_cost_eff),
    ]:
        max_icer, min_icer, median_icer = min_max_median_icer(pairs)
        max_d_abr, min_d_abr, med_d_abr = min_max_median_abr(pairs)
        logger.info(
            f"""
        {label.upper()}:
        Max reduction in weeks spent with bleeding: {abs(max_d_abr):.0f}, ICER: ${max_icer:,.0f}/QALY
        Min reduction in weeks spent with bleeding: {abs(min_d_abr):.0f}, ICER: ${min_icer:,.0f}/QALY
        Median reduction in weeks spent with bleeding: {abs(med_d_abr):.0f}, ICER: ${median_icer:,.0f}/QALY
            """
        )

    # Centroid calculation with safeguard
    m_cost = np.median(delta_costs or [0])
    m_qaly = np.median(delta_qalys or [0])
    centroid = m_cost / m_qaly if m_qaly != 0 else float("inf")
    logger.info(
        f"Median ICER: {'Undefined' if m_qaly == 0 else f'${centroid:,.0f}/QALY'}, Median ΔABR: {np.median(delta_abr or [0]):.2f}"
    )

    def scatter(pairs, label, marker, cmap="viridis", color=None):
        if not pairs:
            logger.warning(f"No data for {label} in ICER scatter")
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

    # Colorbars
    if s_dmd:
        plt.colorbar(s_dmd, ax=icer_ax, label="Dominated ΔABR (weeks)", location="left")
    if s_nce:
        plt.colorbar(s_nce, ax=icer_ax, label="Not Cost-Effective ΔABR (weeks)")
    if s_dom:
        plt.colorbar(s_dom, ax=icer_ax, label="Dominant ΔABR (weeks)")

    # Centroid
    icer_ax.scatter(
        m_qaly, m_cost, color="black", marker="*", s=50, label="Centroid ICER", zorder=5
    )
    icer_ax.annotate(
        f"${centroid:,.0f}/QALY" if m_qaly != 0 else "Undefined",
        (float(m_qaly), float(m_cost)),
        textcoords="offset points",
        xytext=(25, -75),
        ha="center",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="white"),
    )

    # Axes and lines
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

    # Custom legend
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
        ),
    ]
    icer_ax.legend(handles=legend_elements, loc="lower right")
    logger.info("ICER scatter plotted")
    return icer_fig


def plot_icer_histogram(data: DataExtract) -> Figure:
    points = data.icer_pairs  # (dc, dq, da)
    dc = np.array([pair[0] for pair in points])  # Costs
    dq = np.array([pair[1] for pair in points])  # QALYs

    # Define ranges with small buffer
    costs_range = (np.min(dc), np.max(dc))
    qalys_range = (np.min(dq), np.max(dq))
    if np.min(dc) == np.max(dc):
        costs_range = (np.min(dc) - 0.1, np.max(dc) + 0.1)
    if np.min(dq) == np.max(dq):
        qalys_range = (np.min(dq) - 0.1, np.max(dq) + 0.1)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], width_ratios=[1, 3])
    ax_2d = fig.add_subplot(gs[0, 1])  # 2D histogram in top-right
    ax_costs = fig.add_subplot(gs[0, 0], sharey=ax_2d)  # Costs histogram on left
    ax_qalys = fig.add_subplot(gs[1, 1], sharex=ax_2d)  # QALYs histogram below ax_2d
    ax_icer = fig.add_subplot(gs[1, 0])  # ICER histogram bottom left

    # Plot 2D histogram and get bin edges
    hist2d = ax_2d.hist2d(
        x=dq,
        y=dc,
        bins=50,
        cmap="viridis",
        range=[qalys_range, costs_range],
    )

    # Use y-axis bin edges from 2D histogram
    ax_costs.hist(
        dc,
        bins=hist2d[2],  # type: ignore
        range=costs_range,
        orientation="horizontal",
        color="gray",
        edgecolor="black",
    )
    # Use x-axis bin edges from 2D histogram
    ax_qalys.hist(
        dq,
        bins=hist2d[1],  # type: ignore
        range=qalys_range,
        color="gray",
        edgecolor="black",
    )

    # Plot the distribution of the ICERs
    icers = data.categorized[4]
    ax_icer.hist(
        icers,
        range=(-5_000, 50_000),
        bins=25,
        color="gray",
        edgecolor="black",
    )

    ax_2d.set_xlabel("QALYs")
    ax_2d.set_ylabel("Costs")
    ax_costs.set_xlabel("Count")
    ax_costs.set_ylabel("Costs")
    ax_qalys.set_xlabel("QALYs")
    ax_qalys.set_ylabel("Count")
    ax_icer.set_xlabel("ICER")
    ax_icer.set_ylabel("Counts")

    return fig


def plot(
    on_demand_inputs: List[Dict],
    prophylaxis_inputs: List[Dict],
    on_demand_results: Dict,
    prophylaxis_results: Dict,
    n_samples: int,
) -> Dict[str, Figure]:
    """
    Generate all plots and return a dictionary of figures.
    """
    figures: Dict[str, Figure] = {}
    data = extract(
        on_demand_inputs,
        prophylaxis_inputs,
        on_demand_results,
        prophylaxis_results,
        n_samples,
    )

    plot_functions = {
        "factor_consumptions": plot_consumption_vs_abr,
        "factor_histogram": plot_consumption_hist,
        "costs_per_abr": plot_costs_vs_abr,
        "qalys_per_abr": plot_qaly_vs_abr,
        "incremental_cost_effectiveness": plot_icer_scatter,
        "icer_histogram": plot_icer_histogram,
    }

    for key, func in plot_functions.items():
        try:
            fig = func(data)
            figures[key] = fig
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            continue

    return figures
