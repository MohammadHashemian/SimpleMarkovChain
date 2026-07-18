import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from utils.decorators import deprecated
from utils.path_utils import get_project_root

PROJECT_ROOT = get_project_root()


@deprecated("replaced with visualize_matrix()")
def visualize_transition_matrix(
    matrix: np.ndarray, states: list, title: str = "Transition Matrix"
):
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
