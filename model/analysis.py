import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from statsmodels.robust.robust_linear_model import RLMResults


from typing import Tuple


def create_plots(
    on_demand_inputs: list[dict],
    prophylaxis_inputs: list[dict],
    on_demand_results: dict,
    prophylaxis_results: dict,
    n_samples: int,
) -> Tuple[Figure, Figure]:
    """Create separate scatter and histogram plots in individual figures"""
    # Create figure with two subplots
    scatter_fig = plt.figure(figsize=(12, 8))
    scatter_ax: Axes = scatter_fig.add_subplot(1, 1, 1)

    # Extract data for plotting
    on_demand_abr = [inp["abr"] for inp in on_demand_inputs]
    on_demand_ajbr = [inp["ajbr"] for inp in on_demand_inputs]
    on_demand_factors = on_demand_results["total_factors_use"]
    prophylaxis_abr = [inp["abr"] for inp in prophylaxis_inputs]
    prophylaxis_ajbr = [inp["ajbr"] for inp in prophylaxis_inputs]
    prophylaxis_factors = prophylaxis_results["total_factors_use"]

    # --- Plot 1: Scatter plot ---
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

    # Robust regression for On-Demand
    X_od = sm.add_constant(on_demand_abr)
    rlm_od = sm.RLM(on_demand_factors, X_od, M=sm.robust.norms.HuberT())
    od_rlm_results: RLMResults = rlm_od.fit()  # type: ignore
    scatter_ax.plot(
        on_demand_abr,
        od_rlm_results.predict(X_od),
        "b--",
        label=f"On-Demand (robust): slope={od_rlm_results.params[1]:.2f}",
    )

    # Robust regression for Prophylaxis
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

    # --- Plot 2: Histogram ---
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

    return scatter_fig, hist_fig
