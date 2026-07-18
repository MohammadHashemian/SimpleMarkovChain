from matplotlib import pyplot as plt

from utils.decorators import deprecated
from utils.path_utils import get_project_root
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = get_project_root()


@deprecated("replaced with visualize_matrix()")
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


def visualize_abr(abr_values, strategy: str | None = None):
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
    plt.title(
        f"ABR Distribution for {strategy.capitalize() if strategy else 'not-defined'} Strategy"
    )
    plt.xlabel("Annual Bleeding Rate (ABR)")
    plt.ylabel("Frequency")
    plt.savefig(
        PROJECT_ROOT / "outputs" / "figures" / f"abr_distribution_{strategy}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def visualize_matrix(
    matrix: np.ndarray,
    states: list[str],
    inputs,
    sub: str,
    filename: str = "transition_matrix.png",
):
    """
    Visualize CTMC transition matrix + model inputs side-by-side.
    Saves output to /outputs/transitions.
    """

    output_dir = get_project_root() / "outputs" / "transitions" / sub
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    input_dict = {k: v for k, v in vars(inputs).items() if not k.startswith("_")}

    input_text = "\n".join(f"{key}: {value}" for key, value in input_dict.items())

    fig, axes = plt.subplots(
        1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [2.2, 1]}
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        xticklabels=states,
        yticklabels=states,
        ax=axes[0],
    )

    axes[0].set_title("Transition Matrix (CTMC)")

    axes[1].axis("off")
    axes[1].set_title("Model Inputs")

    axes[1].text(
        0.0,
        1.0,
        input_text,
        fontsize=10,
        va="top",
        family="monospace",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return str(output_path)
