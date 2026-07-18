import numpy as np
import matplotlib.pyplot as plt
from analysis.psa.models import ParameterSet
from domain.scenario import Scenario  # NOTE: NOT A GENERIC INTERFACE
from typing import Callable, Any


def plot_trajectories(
    scenarios: list[Scenario],
    default_scenario: ParameterSet,
    trajectory_func: Callable[[int, dict[str, Any]], np.ndarray],
    y_label: str = "Value",
    title_prefix: str = "",
    figsize: tuple | None = None,
    **func_kwargs,
):
    """
    Trajectory plotter.

    Parameters:
    -----------
    scenarios : list[Scenario]
        List of scenarios (with possible overrides)
    default_scenario : ParameterSet
        Base parameters
    trajectory_func : callable
        Function with signature: (cycles: int, params: dict) -> array of shape (cycles,)
        It receives the number of cycles and a dict of resolved parameters.
    y_label : str
        Label for y-axis
    title_prefix : str
        Prefix for subplot titles
    figsize : tuple, optional
        Figure size
    **func_kwargs :
        Extra arguments passed to trajectory_func
    """

    # Extract unique simulation lengths and baseline ages (or any other varying param)
    base_cycles = int(default_scenario.cycles.point())
    base_age = int(default_scenario.baseline_age.point())

    simulation_lengths = {base_cycles}
    baseline_ages = {base_age}

    for scenario in scenarios:
        overrides = scenario.overrides
        if (cycles := overrides.get("cycles")) is not None:
            simulation_lengths.add(int(cycles.point()))
        if (age := overrides.get("baseline_age")) is not None:
            baseline_ages.add(int(age.point()))

    simulation_lengths = sorted(simulation_lengths)
    baseline_ages = sorted(baseline_ages)

    n_rows = len(baseline_ages)
    n_cols = len(simulation_lengths)

    if figsize is None:
        figsize = (4.5 * n_cols, 3.2 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for i, start_age in enumerate(baseline_ages):
        for j, cycles in enumerate(simulation_lengths):
            ax = axes[i, j]

            t = np.arange(cycles)
            age_years = start_age + (t / 52.0)

            # Build parameters for this specific combination
            params = {
                "baseline_age": start_age,
                "cycles": cycles,
                **func_kwargs,
            }

            # Compute trajectory
            y_values = trajectory_func(cycles, params)

            ax.plot(age_years, y_values)

            ax.set_title(f"{title_prefix}Start: {start_age}y | Horizon: {cycles//52}y")
            ax.set_xlabel("Age (years)")
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
