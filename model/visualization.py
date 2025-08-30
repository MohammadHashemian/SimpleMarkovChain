from model.markov_chain import PROJECT_ROOT


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def visualize_transition_matrix(
    matrix: np.ndarray, states: list, title: str = "Transition Matrix"
):
    """
    Visualize a transition matrix as a heatmap.

    Args:
        matrix: Transition probability matrix (square np.ndarray).
        states: List of state names (same length as matrix).
        title: Title for the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".4f",
        xticklabels=states,
        yticklabels=states,
        cmap="Blues",
    )
    plt.title(title)
    plt.xlabel("Next State")
    plt.ylabel("Current State")
    plt.tight_layout()
    plt.savefig(
        PROJECT_ROOT / "outputs" / "figures" / "transitions" / f"{title}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def visualize_abr(abr_values, strategy: str):
    """
    Visualize sampled abr_values and draw histogram of it

    Args:
        abr_values: sampled abr values
        strategy: on_demand or prophylaxis

    Returns:
        None: stores figures at output directory
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(
        abr_values,
        bins=30,
        kde=True,
        color="blue" if strategy == "on_demand" else "green",
    )
    plt.title(f"ABR Distribution for {strategy.capitalize()} Strategy")
    plt.xlabel("Annual Bleeding Rate (ABR)")
    plt.ylabel("Frequency")
    plt.savefig(
        PROJECT_ROOT / "outputs" / "figures" / f"abr_distribution_{strategy}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()