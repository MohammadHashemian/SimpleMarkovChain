from dataclasses import dataclass, replace
from typing import Callable, TypeVar
from model.markov_chain import MarkovChains, MarkovModel
from model.config import ModelConfig, DEFAULT_CONFIG
from model.defined_types import ModelInputAbs
import enlighten
import multiprocessing
import logging

T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True)
class Scenario:
    """Interface to override model parameters at runtime
    title: name of the scenario to be shown in results and progress bars
    n_cycles: number of cycles to run the simulation for
    start_age: age in years at which the simulation starts
    discounting: whether to apply discounting to costs and benefits
    description: optional longer description of the scenario to be shown in results"""

    title: str
    n_cycles: int
    start_age: int
    discounting: bool
    description: str | None = None

    def build_config(self, base: ModelConfig) -> ModelConfig:
        """
        Docstring for build_config

        :param base: Model configuration to use as a base for the scenario configuration
        :type base: ModelConfig
        :param scenario: Scenario configuration to override parameters in the base configuration
        :type scenario: ScenarioConfig
        :return: Updated model configuration with scenario parameters applied
        :rtype: ModelConfig
        """
        econ = base.economics
        if not self.discounting:
            econ = replace(
                econ, discount_rate_costs_annual=0.0, discount_rate_benefits_annual=0.0
            )
        sim = replace(base.simulation, n_cycles=self.n_cycles)
        return replace(base, simulation=sim, economics=econ)


class ModelRunner:
    """
    Summary
    -------
    Class to run Markov model simulations for different scenarios in parallel

    title: name to be shown on progress bar
     worker_inputs: list of keyword arguments to worker_function
     worker_func: a function that should accept the markov_model and input dictionaries
     markov_model: markov model class instance to parallelize within worker function
     config: Model configuration to pass to worker_func as part of worker_kwargs
     scenario_list: list of Scenario instances to run simulations for
    """

    def __init__(
        self,
        title: str,
        markov_model: MarkovModel,
        worker_func: Callable[[MarkovChains, T], tuple[T, U]],
        worker_inputs: list[T],
        scenario_list: list["Scenario"] | None = None,
        base_config: ModelConfig = DEFAULT_CONFIG,
    ):
        self.title = title
        self.markov_model = markov_model
        self.worker_func = worker_func
        self.scenario_list = scenario_list
        self.worker_inputs = worker_inputs
        self.base_config = base_config

    def _update_base_config_for_scenario(self, scenario: Scenario):
        """Updates the base configuration with the scenario parameters for the current scenario"""
        self.base_config = scenario.build_config(self.base_config)

    def run_model_multi_thread(self):
        """
        Summary
        -------
        Gets a worker function with its arguments as a list of dictionaries,
        then uses multiprocessing to pass each dictionary to worker function and returns the results.

        Returns:
            tuple: of ([inputs], [outputs])
        """
        model_inputs = []
        model_outputs = []

        if not self.base_config:
            raise ValueError("ModelConfig is required for parallelized simulations")

        manager = enlighten.get_manager()
        progress_bar: enlighten.Counter = manager.counter(
            total=len(self.worker_inputs),
            desc=f"Simulating {self.title}:",
            unit="simulation",
        )

        def update_bar(_):
            progress_bar.update(incr=1)

        def error_handler(e: BaseException):
            raise ValueError(f"simulation failed {e}")

        def _inject_config(worker_kwargs: T) -> T:
            """Injects the ModelConfig into the worker input dictionary as kwarg 'config'"""
            if isinstance(worker_kwargs, ModelInputAbs):
                return replace(worker_kwargs, config=self.base_config)
            if isinstance(worker_kwargs, dict):
                return {**worker_kwargs, "config": self.base_config}  # type: ignore
            raise ValueError(
                f"Failed to inject config into worker kwargs, expected ModelInputAbs or dict, got {type(worker_kwargs)}"
            )

        # Inject config into each worker input dictionary if it's not already there
        worker_inputs = [_inject_config(kwargs) for kwargs in self.worker_inputs]

        process_count = (
            multiprocessing.cpu_count()
            if len(worker_inputs) >= multiprocessing.cpu_count()
            else len(worker_inputs)
        )
        with multiprocessing.Pool(processes=process_count) as pool:
            async_results = [
                pool.apply_async(
                    func=self.worker_func,
                    args=(self.markov_model, worker_kwargs),
                    callback=update_bar,
                    error_callback=error_handler,
                )
                for worker_kwargs in worker_inputs
            ]
            for res in async_results:
                input_dict, output = res.get()
                model_inputs.append(input_dict)
                model_outputs.append(output)
        manager.stop()
        return model_inputs, model_outputs

    def run_scenarios_multi_thread(self):
        logger = logging.getLogger(__name__)
        results = []
        """Uses run_parallelized_markov_chain to run simulations for each scenario in the scenario_list and collects results"""
        if self.scenario_list is None:
            raise ValueError("No scenarios provided to run_scenarios_multi_thread")
        for scenario in self.scenario_list:
            logger.info(f"Running scenario: {scenario.title}")
            self._update_base_config_for_scenario(scenario)
            output = self.run_model_multi_thread()
            results.append((scenario.title, output))
            logger.info(f"Completed scenario: {scenario.title}, appended results")
        return results
