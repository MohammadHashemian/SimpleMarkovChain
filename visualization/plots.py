from functools import wraps
from typing import Callable, Iterable, Any
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def apply_grid(
    *,
    row_values: Iterable[Any],
    col_values: Iterable[Any],
    row_name: str,
    col_name: str,
    dataframe,
    figsize=(14, 11),
    gridspec_kw=None,
    title_formatter: Callable[[Any, Any], str] | None = None,
    subplot_kwargs=None,
):
    """
    Grid decorator for plotting functions.

    The wrapped function will receive:
        ax, sub, row_value, col_value, i, j

    Example:
        @apply_grid(
            row_values=["early", "lifetime"],
            col_values=["bayesian", "dirichlet"],
            row_name="time_horizon",
            col_name="sampling_method",
            dataframe=df_enhanced,
        )
        def plot(ax, sub, row_value, col_value, i, j):
            ...
    """

    gridspec_kw = gridspec_kw or {
        "wspace": 0.05,
        "hspace": 0.25,
    }

    subplot_kwargs = subplot_kwargs or {}

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):

            row_values_list = list(row_values)
            col_values_list = list(col_values)

            fig, axes = plt.subplots(
                len(row_values_list),
                len(col_values_list),
                figsize=figsize,
                gridspec_kw=gridspec_kw,
                **subplot_kwargs,
            )

            # Normalize axes shape for 1D cases
            if len(row_values_list) == 1 and len(col_values_list) == 1:
                axes = [[axes]]
            elif len(row_values_list) == 1:
                axes = [axes]
            elif len(col_values_list) == 1:
                axes = [[ax] for ax in axes]

            for i, row_value in enumerate(row_values_list):
                for j, col_value in enumerate(col_values_list):

                    ax = axes[i][j]

                    sub = dataframe[
                        (dataframe[row_name] == row_value)
                        & (dataframe[col_name] == col_value)
                    ]

                    func(
                        ax=ax,
                        sub=sub,
                        row_value=row_value,
                        col_value=col_value,
                        i=i,
                        j=j,
                        *args,
                        **kwargs,
                    )

                    if title_formatter:
                        title = title_formatter(row_value, col_value)
                    else:
                        title = f"{row_value} — {col_value}"

                    ax.set_title(title)  # type: ignore

                    ax.grid(True, alpha=0.3)  # type: ignore
                    ax.set_box_aspect(1)  # type: ignore

            return fig, axes

        return wrapper

    return decorator


class OWSAPlotter:

    @staticmethod
    def plot_owsa_icer_tornado(
        summary: pd.DataFrame,
        filter_horizon: str | None = None,
        style: str = "dual_bars",
    ) -> None:
        data = summary.copy()

        if filter_horizon is not None:
            data = data[data["time_horizon"] == filter_horizon]

        # Sort by sensitivity magnitude
        data = data.sort_values("magnitude", ascending=False)

        labels = data["parameter"].astype(str).tolist()

        y = np.arange(len(labels))

        low_vals = data["low_icer_change"].tolist()
        high_vals = data["high_icer_change"].tolist()

        if style == "dual_bars":

            fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5)))

            ax.barh(
                y - 0.2,
                low_vals,
                height=0.35,
                label="Low",
            )

            ax.barh(
                y + 0.2,
                high_vals,
                height=0.35,
                label="High",
            )

            ax.axvline(0, linewidth=1)

            ax.set_yticks(y)
            ax.set_yticklabels(labels)

            ax.set_xlabel("Δ ICER vs Base Case (IRR/QALY)")

            ax.set_title(
                f"OWSA Tornado Diagram — ICER Sensitivity "
                f"({filter_horizon or 'all horizons'})"
            )

            ax.legend()

        elif style == "errorbar":

            fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.35)))

            mid_points = [(l + h) / 2 for l, h in zip(low_vals, high_vals)]

            lower_errors = [abs(m - l) for m, l in zip(mid_points, low_vals)]

            upper_errors = [abs(h - m) for h, m in zip(high_vals, mid_points)]

            ax.errorbar(
                mid_points,
                y,
                xerr=[lower_errors, upper_errors],
                fmt="o",
                markersize=8,
                capsize=5,
                linewidth=2,
                elinewidth=2,
            )

            ax.axvline(0, linewidth=1)

            ax.set_yticks(y)
            ax.set_yticklabels(labels)

            ax.set_xlabel("Δ ICER vs Base Case (IRR/QALY)")

            ax.set_title(
                f"OWSA Sensitivity — ICER " f"({filter_horizon or 'all horizons'})"
            )

        else:
            raise ValueError(
                f"Unknown style: {style}. " "Use 'dual_bars' or 'errorbar'"
            )

        plt.tight_layout()
        plt.show()
