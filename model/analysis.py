from typing import Tuple, List, Dict
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from statsmodels.robust.robust_linear_model import RLMResults
import matplotlib.pyplot as plt
import statsmodels.api as sm


def create_plots(
    on_demand_inputs: List[Dict],
    prophylaxis_inputs: List[Dict],
    on_demand_results: Dict,
    prophylaxis_results: Dict,
    n_samples: int,
):

    # --- Plot 1: Scatter plot (Factor Consumption vs. ABR) ---
    scatter_fig = plt.figure(figsize=(12, 8))
    scatter_ax: Axes = scatter_fig.add_subplot(1, 1, 1)

    # Extract data for plotting
    on_demand_abr = [inp["abr"] for inp in on_demand_inputs]
    on_demand_ajbr = [inp["ajbr"] for inp in on_demand_inputs]
    on_demand_factors = on_demand_results["total_factors_use"]
    on_demand_costs = on_demand_results["total_factors_costs"]
    on_demand_utilities = on_demand_results["QALYS"]
    prophylaxis_abr = [inp["abr"] for inp in prophylaxis_inputs]
    prophylaxis_ajbr = [inp["ajbr"] for inp in prophylaxis_inputs]
    prophylaxis_factors = prophylaxis_results["total_factors_use"]
    prophylaxis_costs = prophylaxis_results["total_factors_costs"]
    prophylaxis_utilities = prophylaxis_results["QALYS"]

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

    # --- Plot 2: Histogram (Factor Consumption Distribution) ---
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

    # --- Plot 3: Scatter plot (Costs vs. ABR) ---
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

    # --- Plot 4: Scatter plot (QALYS vs. ABR) ---
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
    plt.colorbar(scatter1, ax=utility_ax, label="On-Demand QALYs", pad=0.04)

    # Customize axes
    utility_ax.set_xlabel("Annual Bleeding Rate (ABR)")
    utility_ax.set_ylabel("Discounted QALYs")
    utility_ax.set_title("QALYs vs. Annual Bleeding Rate")
    utility_ax.legend()
    utility_ax.grid(True, linestyle="--", alpha=0.7)  # Add grid for readability

    return scatter_fig, hist_fig, cost_fig, utility_fig
