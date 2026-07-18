from dataclasses import dataclass
import gc
import logging
import time
import pandas as pd

from pathlib import Path
from typing import Callable, Literal

from domain.inputs import ModelInput
from domain.scenario import ScenarioBundle
from engine.chains import Chain
from engine.runners import ScenarioRunner
from notebook_tools.dataframe_builders import build_df
from notebook_tools.scenario_helpers import pair_scenarios
from persistence.context import ModelContext
from utils.logging import setup_root_logger


class _Errors:
    MISSING_OUT_DIR = "Output directory %s does not exist. Attempting to create it."
    MISSING_TEMP_DIR = "Temp directory %s does not exist. Cannot use cached temp files."
    USING_CACHED_TEMP = "Using %d cached temp batch parquet files from %s"
    NO_CACHED_TEMP = "No cached temp parquet found in %s, regenerating batches."
    FAILED_BATCH_SAVE = "Failed to write temp batch parquet %s, reason: \n%s"
    NO_RESULTS_COMBINED = "No results available after combining temp batches. Final parquet will not be created."
    FAILED_TO_SAVE_COMBINED_RESULTS = "Failed to save combined results checkpoint %s"
    FAILED_TO_PAIR = "Failed to pair scenarios for final output"
    NO_PAIRS_FOUND = "No scenario pairs found in combined results. Final parquet file list will be empty."
    SKIPPING_PAIR = "Skipping pair %s vs %s: missing arm data"
    FAILED_TO_SAVE_PAIR_RESULTS = "Failed to write Parquet for pair %s vs %s -> %s"
    FAILED_TO_CLEAN_TEMP_DIRECTORY = "Failed to clean up temp directory %s. Manual clean up may be required to free disk space."
    CLEAN_UP_DISABLED = "Clean up disabled for temp directory %s. Manual clean up may be required to free disk space."
    FAILED_TO_BUILD_DF = "Failed to build DataFrame from batch results"


class _Info:
    STARTING_BATCH_RUNNER = (
        "Starting batch runner with %d scenario, batch size: %d, engine: %s"
    )
    SAVED_BATCH = "Saved temp batch %d to %s"
    BATCH_COMPLETE = "Batch %d/%d complete (%d scenarios processed, %d remaining). "
    ELAPSED = "Elapsed=%0.1fs, avg_batch=%0.1fs, est_remaining=%0.1fs"
    COMBINING_TEMP_BATCHES = "Combining %d temp batch files into a DF for pairing"
    SAVED_COMBINED_RESULTS = "Saved combined results checkpoint to %s"
    SAVED_PAIR_RESULTS = "Saved Parquet for pair %s vs %s -> %s"
    SAVED_FINAL_RESULTS = "Saved %d final parquet files to %s"


def batch_generator(bundles: list[ScenarioBundle[ModelInput]], batch_size: int):
    for i in range(0, len(bundles), batch_size):
        yield bundles[i : i + batch_size]


def _safe_name(s: str) -> str:
    return s.replace(" ", "_").replace("/", "_")


def run_scenarios_in_batches(
    bundles: list[ScenarioBundle[ModelInput]],
    context: ModelContext,
    identity_chain: Chain,
    worker_function: Callable,
    batch_size: int,
    output_dir: Path,
    temp_dir: Path,
    options: dict = {"use_cache_temp": False},
    engine: Literal["pathos", "multiprocessing"] = "pathos",
):
    """Run scenario bundles in batches, write per-pair Parquet files, and free memory after each batch."""

    logger = setup_root_logger()
    logger.info(
        _Info.STARTING_BATCH_RUNNER,
        len(bundles),
        batch_size,
        engine,
    )
    use_cached_temp: bool = options.get("use_cached_temp", False)

    if not output_dir.exists():
        logger.info(_Errors.MISSING_OUT_DIR, output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    if not temp_dir.exists():
        logger.info(_Errors.MISSING_TEMP_DIR, temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

    temp_files = []
    if use_cached_temp:
        # Step 0: Load cache from temp storage
        temp_files = sorted(temp_dir.glob("batch_*.parquet"))
        if temp_files:
            logger.info(
                _Errors.USING_CACHED_TEMP,
                len(temp_files),
                temp_dir,
            )
        else:
            logger.info(
                _Errors.NO_CACHED_TEMP,
                temp_dir,
            )

    if not use_cached_temp or not temp_files:
        # Step 1: Process batches and save each to temp storage
        temp_files = []
        total_batches = (len(bundles) + batch_size - 1) // batch_size
        batch_start_time = time.perf_counter()
        processed_scenarios = 0

        for index, batch in enumerate(batch_generator(bundles, batch_size)):
            runner = ScenarioRunner(
                context=context,
                scenario_bundles=batch,
                chain_instance=identity_chain,
                worker_func=worker_function,
                backend=engine,
            )
            batch_results = runner.run_all()

            # convert to DataFrame using existing helper
            try:
                batch_df = build_df(results=batch_results, context=context)
            except Exception as e:
                logger.exception(_Errors.FAILED_TO_BUILD_DF)
                batch_df = pd.DataFrame()

            # save temp batch file
            if not batch_df.empty:
                temp_path = temp_dir / f"batch_{index}.parquet"
                try:
                    batch_df.to_parquet(temp_path, index=False)
                    temp_files.append(temp_path)
                    logger.info(_Info.SAVED_BATCH, index, temp_path)
                except Exception as e:
                    logger.exception(_Errors.FAILED_BATCH_SAVE, temp_path, e.__str__())

            processed_scenarios += len(batch)
            elapsed = time.perf_counter() - batch_start_time
            avg_batch_time = elapsed / (index + 1)
            batches_remaining = total_batches - (index + 1)
            scenarios_remaining = len(bundles) - processed_scenarios
            remaining_time = avg_batch_time * batches_remaining
            logger.info(
                _Info.BATCH_COMPLETE + _Info.ELAPSED,
                index + 1,
                total_batches,
                processed_scenarios,
                scenarios_remaining,
                elapsed,
                avg_batch_time,
                remaining_time,
            )

            # free memory after processing this batch
            try:
                del batch_results
                del batch_df
            except Exception:
                pass
            gc.collect()

    # Step 2: Read all temp files and combine by scenario pair
    logger.info(_Info.COMBINING_TEMP_BATCHES, len(temp_files))
    if temp_files:
        all_results_df = pd.concat(
            [pd.read_parquet(f) for f in temp_files], axis=0, ignore_index=True
        )
    else:
        all_results_df = pd.DataFrame()

    if all_results_df.empty:
        logger.warning(_Errors.NO_RESULTS_COMBINED)
        return []

    # Optional debug output: keep combined data for inspection if needed
    combined_path = output_dir / "all_results_combined.parquet"
    try:
        all_results_df.to_parquet(combined_path, index=False)
        logger.info(_Info.SAVED_COMBINED_RESULTS, combined_path)
    except Exception as e:
        logger.exception(_Errors.FAILED_TO_SAVE_COMBINED_RESULTS, combined_path)

    # Step 3: Group and write per-pair parquet files
    saved_files = []
    try:
        all_pairs = pair_scenarios(all_results_df["scenario"].unique().tolist())
    except Exception as e:
        logger.exception(_Errors.FAILED_TO_PAIR)
        raise

    if not all_pairs:
        logger.warning(_Errors.NO_PAIRS_FOUND)

    for control, intervention in all_pairs:
        control_df = all_results_df[all_results_df["scenario"] == control]
        intervention_df = all_results_df[all_results_df["scenario"] == intervention]

        if control_df.empty or intervention_df.empty:
            logger.warning(_Errors.SKIPPING_PAIR, control, intervention)
            continue

        # combine into a single file with a column indicating arm
        control_df = control_df.copy()
        intervention_df = intervention_df.copy()
        control_df["arm"] = "control"
        intervention_df["arm"] = "intervention"

        combined = pd.concat([control_df, intervention_df], axis=0, ignore_index=True)

        fname = f"{_safe_name(control)}_vs_{_safe_name(intervention)}.parquet"
        path = output_dir / fname
        try:
            combined.to_parquet(path, index=False)
            saved_files.append(path)
            logger.info(_Info.SAVED_PAIR_RESULTS, control, intervention, path)
        except Exception as e:
            logger.exception(_Errors.FAILED_TO_SAVE_PAIR_RESULTS, path)

    # Step 4: Clean up temp files
    try:
        # shutil.rmtree(temp_dir)
        # logger.info("Cleaned up temp directory")
        logger.warning(_Errors.CLEAN_UP_DISABLED, temp_dir)
    except Exception as e:
        logger.exception(_Errors.FAILED_TO_CLEAN_TEMP_DIRECTORY, temp_dir)

    logger.info(_Info.SAVED_FINAL_RESULTS, len(saved_files), output_dir)
    return saved_files
