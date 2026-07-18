from __future__ import annotations

import copy
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any, Generic, Literal, TypeVar

import enlighten

T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True)
class SimulationResult(Generic[T, U]):
    run_id: str
    scenario: str
    worker_id: int
    input_data: T
    output: U


def _get_scenario_name(scenario: Any) -> str:
    return getattr(scenario, "name", str(scenario))


def _worker_entry(args):
    worker_id, input_data, chain, context, worker_func, run_id, scenario = args

    model = copy.deepcopy(chain)
    ctx = context  # immutable → safe

    output = worker_func(
        model,
        input_data,
        scenario,
        ctx,
        run_id=run_id,
        worker_id=worker_id,
    )

    return SimulationResult(
        run_id=run_id,
        scenario=_get_scenario_name(scenario),
        worker_id=worker_id,
        input_data=input_data,
        output=output,
    )


class Runner(Generic[T, U]):
    def __init__(
        self,
        title: str,
        chain_instance: Any,
        context: Any,
        worker_func: Callable,
        worker_inputs: list[T],
        scenario: Any,
        run_id: str | None = None,
    ):
        self.title = title
        self.chain = chain_instance
        self.context = context
        self.worker_func = worker_func
        self.worker_inputs = worker_inputs
        self.scenario = scenario
        self.run_id = run_id or str(uuid.uuid4())

    def _run_with_pool(
        self, PoolClass, mode: Literal["std", "pathos"]
    ) -> list[SimulationResult]:
        n_inputs = len(self.worker_inputs)

        manager = enlighten.get_manager()
        progress = manager.counter(total=n_inputs, desc=self.title, unit="sim")

        args_list = [
            (
                i,
                inp,
                self.chain,
                self.context,
                self.worker_func,
                self.run_id,
                self.scenario,
            )
            for i, inp in enumerate(self.worker_inputs)
        ]

        results: list[SimulationResult] = []

        with PoolClass() as pool:

            if mode == "std":
                iterator = pool.imap_unordered(_worker_entry, args_list)
            elif mode == "pathos":
                iterator = pool.uimap(_worker_entry, args_list)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            for r in iterator:
                results.append(r)
                progress.update()

        manager.stop()
        return results

    def run_multiprocessing(self) -> list[SimulationResult]:
        ctx = get_context("spawn")
        return self._run_with_pool(ctx.Pool, mode="std")

    def run_pathos(self) -> list[SimulationResult]:
        from pathos.multiprocessing import ProcessingPool as Pool

        return self._run_with_pool(Pool, mode="pathos")


class ScenarioRunner(Generic[T, U]):
    def __init__(
        self,
        scenario_bundles: list[Any],
        chain_instance: Any,
        context: Any,
        worker_func: Callable,
        title: str = "Scenario Simulation",
        run_id: str | None = None,
        backend: Literal["multiprocessing", "pathos"] = "multiprocessing",
    ):
        self.bundles = scenario_bundles
        self.chain = chain_instance
        self.context = context
        self.worker_func = worker_func
        self.title = title
        self.run_id = run_id or uuid.uuid4().hex
        self.backend = backend

    def run_all(self) -> list[SimulationResult[T, U]]:
        all_results: list[SimulationResult[T, U]] = []

        for bundle in self.bundles:
            scenario = bundle.scenario
            inputs = bundle.inputs

            runner = Runner(
                title=f"{self.title} | {_get_scenario_name(scenario)}",
                chain_instance=self.chain,
                context=self.context,
                worker_func=self.worker_func,
                worker_inputs=inputs,
                scenario=scenario,
                run_id=self.run_id,
            )

            if self.backend == "pathos":
                results = runner.run_pathos()
            else:
                results = runner.run_multiprocessing()

            all_results.extend(results)

        return all_results
