# from src.data.scarper import fetch_irc_factors
# from src.data.loaders import (
#     load_global_hemophilia_data,
#     process_amarnameh,
#     merge_and_save,
#     load_irc_data,
# )
# from src.processing.distribution_adjuster import adjust_age_distribution
# from cache import ensure_cache_dir, load_cache, save_cache
# from time import time
# from model import constants
# import matplotlib.pyplot as plt
# import model.markov_chain
# import model.analysis
# import numpy as np
from src.utils.logger import get_logger
from pathlib import Path
import typer

PROJECT_ROOT = Path(__file__).parents[0]
logger = get_logger()  # __main__ logger

app = typer.Typer()


@app.command(
    help="Commandline interface no longer supported, use notebooks to import functions as module"
)
def error_message():
    pass


# @app.command(help="Runs loaders and scarpers, then clean and store the results.")
# async def process(execute: bool):
#     if execute:
#         # Loading & Cleaning Amarnameh files
#         (df_agg_1400, df_agg_1399), df_recombinant_1400 = process_amarnameh()
#         logger.info("-" * 64)
#         # Storing cleaned dataframes
#         output_path = PROJECT_ROOT / "data" / "processed" / "factor_viii_analysis.xlsx"
#         output_path.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists
#         merge_and_save(output_path, df_agg_1400, df_agg_1399, df_recombinant_1400)
#         logger.info("-" * 64)
#         # Loading global hemophilia age distribution data
#         df_known_a, df_known_b = load_global_hemophilia_data()
#         logger.info("-" * 64)
#         known_ha = np.array([0.01, 0.12, 0.07, 0.37, 0.19])
#         known_hb = np.array([0.02, 0.10, 0.06, 0.39, 0.19])
#         # Add the unknown portion of patients age to iran known distribution records
#         logger.info("[Hemophilia A][2023]")
#         POPULATION_PROBABILITIES_HA = adjust_age_distribution(df_known_a, known_ha)
#         logger.info("-" * 64)
#         logger.info("[Hemophilia B][2023]")
#         POPULATION_PROBABILITIES_HB = adjust_age_distribution(df_known_b, known_hb)
#         logger.info("-" * 64)
#         logger.warning(
#             "Probably instead of adjusting to mean, should use population pyramid of iran."
#         )
#         logger.info("-" * 64)
#         logger.info("[IRC_FDA][Pricing 2025]")
#         await fetch_irc_factors()
#         load_irc_data(override=False)
#         return True
#     else:
#         logger.warning("Main runner is disable")
#         return False


# # @app.command(help="Runs new Markov model simulation.")
# def markov(
#     n_samples: int = typer.Option(64, "--n-samples", help="Base PSA sample size"),
#     cache: bool = typer.Option(
#         False,
#         "--cache",
#         help="Use cached results if available, otherwise cache new results.",
#     ),
#     force: bool = typer.Option(
#         False,
#         "--force",
#         help="Force recomputation even if cache exists.",
#     ),
#     plot: bool = typer.Option(
#         False, "--plot", help="Stores simulations plots at /outputs/figures/ directory"
#     ),
# ):
#     np.random.seed(42)  # for reproducibility
#     num_steps = constants.LONG_TERM_CYCLE_COUNTS

#     # Validate n_samples
#     if n_samples <= 0:
#         typer.echo("Error: n_samples must be a positive integer")
#         raise typer.Exit(code=1)

#     # Ensure cache directory exists
#     cache_dir = ensure_cache_dir()
#     on_demand_cache_path = cache_dir / "on_demand.pkl"
#     prophylaxis_cache_path = cache_dir / "prophylaxis.pkl"

#     # Load transition matrix
#     initial_state, states, on_demand_transition = model.markov_chain.load_matrix(
#         io=PROJECT_ROOT / "data" / "Transitions.xlsx",
#         sheet_name="on_demand",
#     )

#     def run_simulation(
#         sim_func,
#         cache_path: Path,
#         states,
#         start_state,
#         steps: int,
#         n_samples: int,
#         use_cache: bool,
#         force_recompute: bool,
#         sim_name: str,
#     ):
#         """Run or load simulation results with caching."""
#         start_time = time()

#         if use_cache and not force_recompute and cache_path.exists():
#             cache_data = load_cache(cache_path, n_samples, steps)
#             if cache_data:
#                 inputs, outputs = cache_data
#                 typer.echo(
#                     f"{sim_name} loaded from cache in {(time() - start_time):.0f} seconds."
#                 )
#                 return inputs, outputs

#         # Run simulation if no cache or force recompute
#         inputs, outputs = sim_func(
#             states=states, start_state=start_state, steps=steps, n_samples=n_samples
#         )
#         if use_cache:
#             save_cache(cache_path, inputs, outputs, n_samples, steps)

#         typer.echo(
#             f"{sim_name} simulation completed in {(time() - start_time):.0f} seconds."
#         )
#         return inputs, outputs

#     # Run On-Demand simulation
#     on_demand_inputs, on_demand_outputs = run_simulation(
#         sim_func=model.markov_chain.on_demand_psa,
#         cache_path=on_demand_cache_path,
#         states=states,
#         start_state=initial_state,
#         steps=num_steps,
#         n_samples=n_samples,
#         use_cache=cache,
#         force_recompute=force,
#         sim_name="On-Demand",
#     )

#     # Output On-Demand results
#     typer.echo(f"Number of samples: {len(on_demand_inputs)}")
#     typer.echo(f"Sample input example: {on_demand_inputs[0]}")
#     typer.echo(
#         f"Median total factor use: {np.median(on_demand_outputs['total_factors_use']):,.0f}"
#     )
#     typer.echo(
#         f"Median annual factor consumption: {np.median(on_demand_outputs['annual_factor_consumption']):,.0f}"
#     )
#     typer.echo(
#         f"Median annual factor costs: ${np.median(on_demand_outputs['total_factors_costs']):,.0f}"
#     )
#     typer.echo(f"Median QALYS: {np.median(on_demand_outputs['QALYS']):.2f}")

#     # Run Prophylaxis simulation
#     prophylaxis_inputs, prophylaxis_outputs = run_simulation(
#         sim_func=model.markov_chain.prophylaxis_psa,
#         cache_path=prophylaxis_cache_path,
#         states=states,
#         start_state=initial_state,
#         steps=num_steps,
#         n_samples=n_samples,
#         use_cache=cache,
#         force_recompute=force,
#         sim_name="Prophylaxis",
#     )

#     # Output Prophylaxis results
#     typer.echo(f"Number of samples: {len(prophylaxis_inputs)}")
#     typer.echo(f"Sample input example: {prophylaxis_inputs[0]}")
#     typer.echo(
#         f"Median total factor use (Prophylaxis): {np.median(prophylaxis_outputs['total_factors_use']):,.0f}"
#     )
#     typer.echo(
#         f"Median annual factor consumption: {np.median(prophylaxis_outputs['annual_factor_consumption']):,.0f}"
#     )
#     typer.echo(
#         f"Median annual factor costs: ${np.median(prophylaxis_outputs['total_factors_costs']):,.0f}"
#     )
#     typer.echo(f"Median QALYS: {np.median(prophylaxis_outputs['QALYS']):.2f}")

#     # Generate and save plots
#     if plot:
#         suppress_matplotlib_font_manager_logger()
#         plots = model.analysis.plot(
#             on_demand_inputs,
#             prophylaxis_inputs,
#             on_demand_outputs,
#             prophylaxis_outputs,
#             n_samples=n_samples,
#         )
#         save_dir = PROJECT_ROOT / "outputs" / "figures"
#         save_dir.mkdir(parents=True, exist_ok=True)
#         plt.tight_layout()
#         for name, fig in plots.items():
#             if not fig:
#                 raise TypeError(
#                     f"Figure name {name} is not defined, returned type: {type(fig)}"
#                 )
#             fig.savefig(save_dir / f"{name}.png", dpi=300, bbox_inches="tight")
#             plt.close(fig)
#         typer.echo(f"Plots saved to {save_dir}")


if __name__ == "__main__":
    logger.warning(
        "Commandline interface no longer supported, use notebooks to import functions as module"
    )
    app()
