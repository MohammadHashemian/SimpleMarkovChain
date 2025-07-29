from src.utils.logger import get_logger, suppress_matplotlib_debug
from src.data.scarper import fetch_irc_factors
from src.data.loaders import (
    load_global_hemophilia_data,
    process_amarnameh,
    merge_and_save,
    load_irc_data,
)
from src.processing.distribution_adjuster import adjust_age_distribution
from model_dep import simulation
from pathlib import Path
from time import time
import model.markov
import model.analysis
import model.constants
import matplotlib.pyplot as plt
import numpy as np
import typer

PROJECT_ROOT = Path(__file__).parents[0]
logger = get_logger()

app = typer.Typer()


@app.command(help="Runs loaders and scarpers, then clean and store the results.")
async def process(execute: bool):
    if execute:
        # Loading & Cleaning Amarnameh files
        (df_agg_1400, df_agg_1399), df_recombinant_1400 = process_amarnameh()
        logger.info("-" * 64)
        # Storing cleaned dataframes
        output_path = PROJECT_ROOT / "data" / "processed" / "factor_viii_analysis.xlsx"
        output_path.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists
        merge_and_save(output_path, df_agg_1400, df_agg_1399, df_recombinant_1400)
        logger.info("-" * 64)
        # Loading global hemophilia age distribution data
        df_known_a, df_known_b = load_global_hemophilia_data()
        logger.info("-" * 64)
        known_ha = np.array([0.01, 0.12, 0.07, 0.37, 0.19])
        known_hb = np.array([0.02, 0.10, 0.06, 0.39, 0.19])
        # Add the unknown portion of patients age to iran known distribution records
        logger.info("[Hemophilia A][2023]")
        POPULATION_PROBABILITIES_HA = adjust_age_distribution(df_known_a, known_ha)
        logger.info("-" * 64)
        logger.info("[Hemophilia B][2023]")
        POPULATION_PROBABILITIES_HB = adjust_age_distribution(df_known_b, known_hb)
        logger.info("-" * 64)
        logger.warning(
            "Probably instead of adjusting to mean, should use population pyramid of iran."
        )
        logger.info("-" * 64)
        logger.info("[IRC_FDA][Pricing 2025]")
        await fetch_irc_factors()
        load_irc_data(override=False)
        return True
    else:
        logger.warning("Main runner is disable")
        return False


@app.command(help="Runs new markov model simulation.")
def markov(
    n_samples: int = typer.Option(64, "--n-samples", help="Number of samples for PSA."),
    plot: bool = typer.Option(False, "--plot", help="Generate the plots of results."),
):
    num_steps = model.constants.NUM_CYCLES
    # np.random.seed(42)  # For reproducibility

    # Validate n_samples
    if n_samples <= 0:
        typer.echo("Error: n_samples must be a positive integer.")
        raise typer.Exit(code=1)

    # On_Demand
    initial_state, states, on_demand_transition = model.markov.load_transition_matrix(
        io=PROJECT_ROOT / "data" / "Transitions.xlsx",
        sheet_name="on_demand",
    )
    new = time()
    on_demand_inputs, on_demand_results = model.markov.on_demand_psa(
        states=states, start_state=initial_state, steps=num_steps, n_samples=n_samples
    )
    exc = time()
    typer.echo(f"On_Demand Simulation completed in {(exc - new):.0f} seconds.")
    typer.echo(f"Number of samples: {len(on_demand_inputs)}")
    typer.echo(f"Sample input example: {on_demand_inputs[0]}")
    typer.echo(
        f"Mean total factor use: {np.mean(on_demand_results['total_factors_use']):.0f}"
    )
    typer.echo(
        f"Mean annual factor consumption: {np.mean(on_demand_results['annual_factor_consumption']):.0f}"
    )
    typer.echo(
        f"Mean annual factor discounted costs: {np.mean(on_demand_results['total_factors_costs']):.0f}$, PPP"
    )
    typer.echo(f"Mean discounted QALYS {np.mean(on_demand_results['QALYS']):.0f}")

    # Prophylaxis
    new = time()
    prophylaxis_inputs, prophylaxis_results = model.markov.prophylaxis_psa(
        states=states, start_state=initial_state, steps=num_steps, n_samples=n_samples
    )
    exc = time()
    typer.echo(f"Prophylaxis Simulation completed in {(exc - new):.0f} seconds.")
    typer.echo(f"Number of samples: {len(prophylaxis_inputs)}")
    typer.echo(f"Sample input example: {prophylaxis_inputs[0]}")
    typer.echo(
        f"Mean total factor use (Prophylaxis): {np.mean(prophylaxis_results['total_factors_use']):.0f}"
    )
    typer.echo(
        f"Mean annual factor consumption: {np.mean(prophylaxis_results['annual_factor_consumption']):.0f}"
    )
    typer.echo(
        f"Mean annual factor discounted costs: {np.mean(prophylaxis_results['total_factors_costs']):.0f}$, PPP"
    )
    typer.echo(f"Mean discounted QALYS: {np.mean(prophylaxis_results['QALYS']):.2f}")

    if plot:
        suppress_matplotlib_debug()
        scatter, hist, cost_fig, utility_fig = model.analysis.create_plots(
            on_demand_inputs,
            prophylaxis_inputs,
            on_demand_results,
            prophylaxis_results,
            n_samples=n_samples,
        )
        # Save plot
        save_dir = PROJECT_ROOT / "outputs" / "figures"
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        scatter.savefig(
            save_dir / "factor_consumption_scatter.png", dpi=300, bbox_inches="tight"
        )
        hist.savefig(
            save_dir / "factor_consumption_histogram.png", dpi=300, bbox_inches="tight"
        )
        cost_fig.savefig(
            save_dir / "factor_costs_scatter.png", dpi=300, bbox_inches="tight"
        )
        utility_fig.savefig(
            save_dir / "utility_over_abr_scatter.png", dpi=300, bbox_inches="tight"
        )

        plt.close(scatter)
        plt.close(hist)
        typer.echo(f"Plots saved to {save_dir}")


if __name__ == "__main__":
    app()
