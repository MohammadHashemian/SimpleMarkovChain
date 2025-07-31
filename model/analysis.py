from typing import List, Dict
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from statsmodels.robust.robust_linear_model import RLMResults
from src.utils.logger import get_logger
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np


logger = get_logger()


# --- Plot 1: Scatter plot (Factor Consumption vs. ABR) ---
def plot_consumption_vs_abr(*args):
    (
        on_demand_inputs,
        prophylaxis_inputs,
        on_demand_results,
        prophylaxis_results,
        n_samples,
    ) = args
    scatter_fig = plt.figure(figsize=(12, 8))
    scatter_ax: Axes = scatter_fig.add_subplot(1, 1, 1)

    # Extract data for plotting
    on_demand_abr = [inp["abr"] for inp in on_demand_inputs]
    on_demand_ajbr = [inp["ajbr"] for inp in on_demand_inputs]
    on_demand_factors = on_demand_results["total_factors_use"]
    prophylaxis_factors = prophylaxis_results["total_factors_use"]
    prophylaxis_abr = [inp["abr"] for inp in prophylaxis_inputs]
    prophylaxis_ajbr = [inp["ajbr"] for inp in prophylaxis_inputs]

    # Scatter plot for factor consumption
    scatter1 = scatter_ax.scatter(
        on_demand_abr,
        on_demand_factors,
        c=on_demand_ajbr,
        cmap="viridis",
        label="On-Demand",
        alpha=0.6,
        s=50,
    )
    scatter2 = scatter_ax.scatter(
        prophylaxis_abr,
        prophylaxis_factors,
        c=prophylaxis_ajbr,
        cmap="plasma",
        label="Prophylaxis",
        marker="^",
        alpha=0.6,
        s=50,
    )

    # Robust regression for On-Demand (factor consumption)
    X_od = sm.add_constant(on_demand_abr)
    rlm_od = sm.RLM(on_demand_factors, X_od, M=sm.robust.norms.HuberT())
    od_rlm_results: RLMResults = rlm_od.fit()  # type: ignore
    scatter_ax.plot(
        on_demand_abr,
        od_rlm_results.predict(X_od),
        "b--",
        label=f"On-Demand (robust): slope={od_rlm_results.params[1]:.2f}",
    )

    # Robust regression for Prophylaxis (factor consumption)
    X_pro = sm.add_constant(prophylaxis_abr)
    rlm_pro = sm.RLM(prophylaxis_factors, X_pro, M=sm.robust.norms.HuberT())
    prophylaxis_rlm_results: RLMResults = rlm_pro.fit()  # type: ignore
    scatter_ax.plot(
        prophylaxis_abr,
        prophylaxis_rlm_results.predict(X_pro),
        "r--",
        label=f"Prophylaxis (robust): slope={prophylaxis_rlm_results.params[1]:.2f}",
    )

    scatter_ax.set_xlabel("Annual Bleeding Rate (ABR)")
    scatter_ax.set_ylabel("Total Factor Consumption (Units)")
    scatter_ax.set_title("Factor Consumption vs. ABR")
    scatter_ax.legend()
    scatter_ax.grid(True, alpha=0.3)
    scatter_fig.colorbar(scatter1, ax=scatter_ax, label="AJBR (On-Demand)")

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


# --- Plot 3: Scatter plot (Costs vs. ABR) ---
def plot_costs_vs_abr(*args):
    (
        on_demand_inputs,
        prophylaxis_inputs,
        on_demand_results,
        prophylaxis_results,
        n_samples,
    ) = args

    # Extract data for plotting
    on_demand_abr = [inp["abr"] for inp in on_demand_inputs]
    on_demand_ajbr = [inp["ajbr"] for inp in on_demand_inputs]
    on_demand_costs = on_demand_results["total_factors_costs"]
    prophylaxis_abr = [inp["abr"] for inp in prophylaxis_inputs]
    prophylaxis_ajbr = [inp["ajbr"] for inp in prophylaxis_inputs]
    prophylaxis_costs = prophylaxis_results["total_factors_costs"]

    cost_fig = plt.figure(figsize=(12, 8))
    cost_ax: Axes = cost_fig.add_subplot(1, 1, 1)

    # Scatter plot for costs
    cost_scatter1 = cost_ax.scatter(
        on_demand_abr,
        on_demand_costs,
        c=on_demand_ajbr,
        cmap="viridis",
        label="On-Demand",
        alpha=0.6,
        s=50,
    )
    cost_scatter2 = cost_ax.scatter(
        prophylaxis_abr,
        prophylaxis_costs,
        c=prophylaxis_ajbr,
        cmap="plasma",
        label="Prophylaxis",
        marker="^",
        alpha=0.6,
        s=50,
    )

    # Robust regression for On-Demand (costs)
    X_od_cost = sm.add_constant(on_demand_abr)
    rlm_od_cost = sm.RLM(on_demand_costs, X_od_cost, M=sm.robust.norms.HuberT())
    od_rlm_cost_results: RLMResults = rlm_od_cost.fit()  # type: ignore
    cost_ax.plot(
        on_demand_abr,
        od_rlm_cost_results.predict(X_od_cost),
        "b--",
        label=f"On-Demand (robust): slope={od_rlm_cost_results.params[1]:.2f}",
    )

    # Robust regression for Prophylaxis (costs)
    X_pro_cost = sm.add_constant(prophylaxis_abr)
    rlm_pro_cost = sm.RLM(prophylaxis_costs, X_pro_cost, M=sm.robust.norms.HuberT())
    pro_rlm_cost_results: RLMResults = rlm_pro_cost.fit()  # type: ignore
    cost_ax.plot(
        prophylaxis_abr,
        pro_rlm_cost_results.predict(X_pro_cost),
        "r--",
        label=f"Prophylaxis (robust): slope={pro_rlm_cost_results.params[1]:.2f}",
    )

    cost_ax.set_xlabel("Annual Bleeding Rate (ABR)")
    cost_ax.set_ylabel("Mean Annual Factor Cost (USD)")
    cost_ax.set_title("Factor Costs vs. ABR")
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


# Utility to plot ICERs plots
WTP = 20_000


def prepare_icer(*args):
    (
        on_demand_inputs,
        prophylaxis_inputs,
        on_demand_results,
        prophylaxis_results,
        n_samples,
    ) = args
    # Extract data for plotting
    on_demand_abr = [inp["abr"] for inp in on_demand_inputs]
    prophylaxis_abr = [inp["abr"] for inp in prophylaxis_inputs]
    on_demand_costs = on_demand_results["total_factors_costs"]
    prophylaxis_costs = prophylaxis_results["total_factors_costs"]
    on_demand_utilities = on_demand_results["QALYS"]
    prophylaxis_utilities = prophylaxis_results["QALYS"]

    # Prepare (Cost, QALY, ABR) pairs
    on_demand_pair = [
        (on_demand_costs[i], q, on_demand_abr[i])
        for i, q in enumerate(on_demand_utilities)
    ]
    prophylaxis_pair = [
        (prophylaxis_costs[i], q, prophylaxis_abr[i])
        for i, q in enumerate(prophylaxis_utilities)
    ]

    # Removes Increase in ABR ICER, Impossible transition
    icer_pairs = [
        (pp[0] - op[0], pp[1] - op[1], pp[2] - op[2])
        for op in on_demand_pair
        for pp in prophylaxis_pair
        if not (pp[2] - op[2]) > 0
    ]

    # Categorize ICERs
    wtp = 20_000
    dominant, dominated, cost_eff, not_cost_eff, icers = [], [], [], [], []

    for dc, dq, da in icer_pairs:
        if da > 0:
            logger.error("ICER calculation with positive delta ABR is prohibited")
        if dq < 0:  # Dominated: worse outcome, higher cost
            pair = (float("inf"), (dc, dq, da))  # Use inf for dominated ICERs
            dominated.append(pair)
            continue
        if dq == 0:  # Handle zero QALYs to avoid division by zero
            pair = (float("inf"), (dc, dq, da))
            not_cost_eff.append(pair)
            continue
        icer = dc / dq
        icers.append(icer)
        pair = (icer, (dc, dq, da))
        if dc < 0 and dq > 0:  # Dominant: cost-saving, better outcome
            dominant.append(pair)
        elif icer <= wtp:  # Cost-effective
            cost_eff.append(pair)
        else:  # Not cost-effective
            not_cost_eff.append(pair)
    logger.info("Categorized ICERS pairs:")
    logger.info(f"Dominant: {len(dominant)}, Dominated: {len(dominated)}")
    logger.info(
        f"Cost-effective: {len(not_cost_eff)}, Not Cost-effective: {len(cost_eff)}"
    )
    try:
        assert np.isclose(
            len(icer_pairs),
            len(dominant) + len(dominated) + len(cost_eff) + len(not_cost_eff),
        )
        logger.info("Categorization asserted")
    except AssertionError:
        logger.warning("Data lost during categorization")
    return icer_pairs, (dominant, dominated, cost_eff, not_cost_eff, icers)


# --- Plot 4: Scatter plot (Costs vs. QALYs) ---
def plot_icer_scatter(*args):
    icer_fig = plt.figure(figsize=(20, 10))
    icer_ax = icer_fig.add_subplot(1, 1, 1)

    icer_pairs, (dominant, dominated, cost_eff, not_cost_eff, icers) = prepare_icer(
        *args
    )

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
    icer_ax.plot(x_rng, x_rng * WTP, "k--", alpha=0.8, label=f"WTP: ${WTP:,}/QALY")
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
            [0], [0], linestyle="--", color="black", label=f"WTP: ${WTP:,}/QALY"
        ),  # WTP line
    ]

    icer_ax.legend(handles=legend_elements, loc="lower right")
    return icer_fig


def plot_icer_histogram(*args):
    icer_fig = plt.figure(figsize=(12, 8))
    gs = icer_fig.add_gridspec(1, 2)

    icer_pairs, (dominant, dominated, cost_eff, not_cost_eff, icers) = prepare_icer(
        *args
    )

    dmd_ax = icer_fig.add_subplot(gs[0])
    # TODO:
    # Dominated histogram ABR distribution here

    # --- Histogram of ICERs ---
    hist_ax = icer_fig.add_subplot(gs[1])

    hist_groups = [
        ([icer for icer, _ in dominant], "green", "Dominant"),
        ([icer for icer, _ in cost_eff], "blue", "Cost-Effective"),
        ([icer for icer, _ in not_cost_eff], "red", "Not Cost-Effective"),
        # Exclude dominated from histogram since ICERs are undefined (inf)
    ]

    for icers_list, color, label in hist_groups:
        if icers_list:  # Only plot if list is not empty
            hist_ax.hist(
                icers_list,
                bins=20,
                range=(-50000, 50000),
                color=color,
                alpha=0.4,
                label=label,
            )

    # Add frequency of dominated points as text annotation
    hist_ax.text(
        0.05,
        0.5,  # Left center: 5% from left, 50% from bottom
        f"Dominated: {len(dominated)} points (ICER undefined)",
        transform=hist_ax.transAxes,
        fontsize=9,
        verticalalignment="center",
        horizontalalignment="left",
        rotation="vertical",
        bbox=dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="white"),
    )

    # Add WTP line
    hist_ax.axvline(
        WTP,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"WTP: ${WTP:,}/QALY",
    )

    hist_ax.set_xlabel("ICER ($/QALY)")
    hist_ax.set_ylabel("Frequency")
    hist_ax.set_title("ICER Distribution")
    hist_ax.grid(True, linestyle="--", alpha=0.7, axis="x")
    hist_ax.legend(loc="upper left")
    return icer_fig


# Plot all results
def plot(
    on_demand_inputs: List[Dict],
    prophylaxis_inputs: List[Dict],
    on_demand_results: Dict,
    prophylaxis_results: Dict,
    n_samples: int,
) -> dict[str, Figure]:
    figures = {}
    plot_functions = {
        "factor_consumption": plot_consumption_vs_abr,
        "factor_histogram": plot_consumption_hist,
        "costs_vs_abr": plot_costs_vs_abr,
        "qalys_vs_abr": plot_qaly_vs_abr,
        "costs_vs_qalys": plot_icer_scatter,
        "icer_histogram": plot_icer_histogram,
    }
    for key, func in plot_functions.items():
        fig = func(
            on_demand_inputs,
            prophylaxis_inputs,
            on_demand_results,
            prophylaxis_results,
            n_samples,
        )
        figures[key] = fig
    return figures
