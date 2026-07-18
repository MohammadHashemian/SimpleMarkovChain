from utils.math import cal_body_weight
from utils.path_utils import get_project_root
import matplotlib.pyplot as plt
import numpy as np


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
    plt.savefig(get_project_root() / "outputs" / "figures" / "body_weight.png")
