<p align="center">
  <img src="logo-light.png" width="500" alt="Hemophilia Markov Model logo" />
</p>

<h1 align="center">🩸 Hemophilia Cost-Effectiveness Markov Model</h1>

<p align="center">
  A discrete-time Markov chain framework for health-economic modeling of hemophilia interventions, with full Probabilistic Sensitivity Analysis (PSA).
</p>

<p align="center">
  <a href="https://github.com/MohammadHashemian/SimpleMarkovChain/actions"><img src="https://github.com/MohammadHashemian/SimpleMarkovChain/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="#"><img src="https://img.shields.io/badge/tests-214%20passed-brightgreen" alt="tests" /></a>
  <a href="#"><img src="https://img.shields.io/badge/coverage-54%25-yellow" alt="coverage" /></a>
  <a href="#"><img src="https://img.shields.io/badge/stability-stable-brightgreen" alt="stability" /></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.11%20%7C%203.14-blue" alt="python" /></a>
  <a href="https://www.gnu.org/licenses/mit.html"><img src="https://img.shields.io/badge/license-MIT-blue" alt="license" /></a>
</p>

---

## ✨ Overview

This project provides a **typed, reproducible, and extensible** discrete-time Markov chain (DTMC) engine purpose-built for **health-economic evaluation** of hemophilia treatments. It couples a robust simulation engine with rigorous uncertainty quantification — including **Probabilistic Sensitivity Analysis (PSA)**, **One-Way Sensitivity Analysis (OWSA)**, and **Bayesian meta-analysis** of clinical inputs.

The modeling logic is shipped as an importable Python package, while all analyses are conducted through well-documented **Jupyter notebooks** so every result is traceable end-to-end.

### 🔑 Key Features

- **⚙️ Flexible transition engine** — DTMC, CTMC, hybrid, and independent-hazard generators with automatic row-normalization and absorbing-state handling.
- **🚀 Vectorized batch engine** — Run thousands of structurally identical simulations in a single numpy sweep via `BatchMarkovChain` and `worker_function_batch`. Reduces a 10k-iter PSA from hours to minutes (12–17× speedup, see [⚡ Performance](#-performance)).
- **🎯 Domain-rich rewards** — Pettersson score, bleeding events, factor consumption, utility weights, and mortality modifiers — with both scalar and vectorized implementations sharing the same semantics.
- **📊 Uncertainty quantification** — PSA (Monte Carlo), OWSA (tornado diagrams), CEAC, and EVPI.
- **🔬 Bayesian foundation** — PyMC-driven posterior sampling and convergence diagnostics (R-hat, ESS, divergences).
- **🧱 Typed IO boundary** — Pydantic schemas validate every external input before it touches the engine.
- **🧪 214 passing tests** spanning the engine, domain, analysis, and persistence layers.

---

## 📂 Directory Layout

| Component | Location | Responsibility |
|---|---|---|
| Input data (JSON) | `data/` | Clinical parameters, costs, utilities, mortality tables, simulation config |
| Pydantic schemas | `persistence/schemas/` | Typed input/output validation models |
| Data loaders | `persistence/loaders.py` | Reading and parsing JSON data sources |
| Model context | `persistence/context.py` | Aggregates all loaded data into a single `ModelContext` |
| Markov engine (scalar) | `engine/chains.py` | `MarkovChains` + `Chain` — per-iteration DTMC runtime |
| Markov engine (vectorized) | `engine/vectorized.py` | `BatchMarkovChain` — n_iters in lockstep, single numpy sweep per step |
| Runners | `engine/runners.py` | `Runner`, `ScenarioRunner` — process pools, chunked IPC, batch dispatch |
| Domain logic | `domain/` | Health states, regime enums, scenarios, reward functions |
| Clinical rewards | `domain/rewards/` | Pettersson score, bleeding events, factor consumption, utility weights, mortality modifiers (scalar) |
| Vectorized rewards | `domain/rewards/hemophilia_vectorized.py` | Per-iter `weight`, `event_count`, `pettersson_score`, `consumption`, `utility` (numpy ufuncs) |
| Batch worker | `domain/worker.py` | `worker_function` (scalar) and `worker_function_batch` (vectorized) |
| PSA distributions | `analysis/psa/` | Sampling distributions, parameter uncertainty, Monte Carlo drivers |
| Economic analysis | `analysis/` | ICER, QALY aggregation, cost-effectiveness computations |
| General utilities | `utils/` | Pure helper functions (math, transformations, decorators, path utilities) |
| Visualization | `visualization/` | Plotting functions and result visualization utilities |
| Notebook tools | `notebook_tools/` | Helper modules shared across notebooks (incl. `run_scenarios_in_batches`) |
| Notebooks | `notebooks/` | Analysis notebooks (see below) |
| Tests | `tests/` | Pytest test suite (incl. `test_vectorized.py` for the batch engine) |
| Outputs | `outputs/` | Generated figures, logs, and simulation results |

---

## 📓 Notebooks

| Notebook | Purpose |
|---|---|
| `00_poisson_mass_functions.ipynb` | Poisson mass function validation for bleeding event distributions |
| `01_preprocessing.ipynb` | Data preprocessing and parameter derivation |
| `01b_mortality_iran.ipynb` | UN WPP 2024 mortality reconstruction for Iran, derivation of the `data/mortality_iran.json` table |
| `02_meta_analysis.ipynb` | Bayesian meta-analysis of clinical inputs |
| `03_psa_simulation.ipynb` | Probabilistic sensitivity analysis (10,000 iterations, vectorized batch engine) |
| `04_owsa_simulation.ipynb` | One-way sensitivity analysis simulation |
| `05_psa_analysis.ipynb` | PSA result analysis (CEAC, EVPI, scatter plots) |
| `07_owsa_analysis.ipynb` | OWSA result analysis (tornado diagrams) |

> **Reproducibility note.** Both PSA (`03`) and OWSA (`04`) notebooks derive per-scenario RNG seeds from the env seed in `data/simulation.json` using `utils.stable_hash` (a CRC-32–based, cross-process-stable hash). Do not use Python's built-in `hash()` for this — it is randomized per process by `PYTHONHASHSEED` and will silently break reproducibility.

---

## ⚡ Performance

The codebase ships with **two execution paths** for the Markov simulation:

| Path | Module | Use case | Behavior |
|---|---|---|---|
| **Scalar** | `engine/chains.py` — `MarkovChains` | Small-scale runs, custom reward logic, debugging | One Python iteration per (iter, step); full reward-fn flexibility |
| **Vectorized** | `engine/vectorized.py` — `BatchMarkovChain` | PSA, OWSA, any batch of homogeneous simulations | Stacks `n_iters` per-iter state into `(n_iters, n_states)` arrays; one numpy op per step across all iters |

Both paths share the **same domain semantics** — the vectorized reward functions in `domain/rewards/hemophilia_vectorized.py` mirror the scalar ones in `domain/rewards/hemophilia.py` bit-for-bit (within sampling noise), so results are statistically equivalent.

### Measured speedup

Head-to-head on a real hemophilia 8-state chain with the full reward + mortality-modifier workload:

| Scenario | Steps | Scalar (per iter) | Vectorized (per iter) | Speedup |
|---|---|---|---|---|
| Early (10 yrs, weekly) | 520 | 37.8 ms | 2.30 ms | **16.5×** |
| Lifetime (98 yrs, weekly) | 5096 | 85.2 ms | 6.66 ms | **12.8×** |

For the Heavy PSA workload (10,000 iters across 18 scenarios), this drops end-to-end runtime from **hours to ~20 minutes** on a single core, with linear scaling on multi-core pools.

### How to use the vectorized path

In any notebook or script, pass `engine="batch"` to the scenario runner and supply the `batch_worker_function`:

```python
from domain.worker import worker_function, worker_function_batch
from notebook_tools.scenario_runner import run_scenarios_in_batches

run_scenarios_in_batches(
    bundles=bundles,
    context=context,
    identity_chain=identity_chain,
    batch_size=4,
    engine="batch",                          # <- was: "pathos" or "multiprocessing"
    worker_function=worker_function,         # kept for backwards compat
    batch_worker_function=worker_function_batch,  # <- new
    output_dir=...,
    temp_dir=...,
)
```

Internally, the runner creates a per-scenario `enlighten` progress bar that reports `X/n_iters iter` with an ETA and throughput, e.g.:

```
Batch Simulation | early on-demand bayesian | 10000 iters  30%|███   | 3000/10000 iter [00:06<00:14, 500.00 iter/s]
```

### Where the speedup comes from

- **Step loop is O(steps), not O(n_iters × steps).** The hot inner loop runs once per simulated step, doing one numpy op over `(n_iters, n_states)` arrays.
- **Vectorized absorbing-state check** via a precomputed boolean mask per chain.
- **Vectorized mortality modifier** — only applied at year boundaries (51 of every 52 weeks are a no-op).
- **Vectorized categorical sampling** via cumsum + uniform draws (`argmax(u < cumsum)`).
- **Vectorized reward functions** — `weight`, `event_count`, `pettersson_score`, `consumption`, `utility` all operate as numpy ufuncs on `(n_iters,)` arrays.
- **Plus** the scalar-path micro-optimizations in `MarkovChains.walk` and `Runner._run_with_pool` (cached worker kwargs, pre-resolved absorbing mask, no-conditions / NoOp-modifier fast paths, `chunksize` on `imap_unordered`, no `copy.deepcopy` in the worker entrypoint).

### Adding a new vectorized reward function

The `BatchMarkovChain.walk_batch` step loop calls store and reward functions with this signature:

```python
def fn(step: int,
       state_idx: np.ndarray,         # (n_iters,) current state index per iter
       store_arrays: dict,            # previously-computed store values
       shared_kwargs: dict,           # per-batch constants
       rng: np.random.Generator) -> np.ndarray:  # returns (n_iters,) array
```

To add a new vectorized reward, write a function with this signature and pass it via the `store_funcs` / `reward_funcs` dict to `walk_batch` (or extend the `VECTORIZED_STORE_FUNCS` / `VECTORIZED_REWARD_FUNCS` registries in `domain/rewards/hemophilia_vectorized.py`).

### Progress callback for custom runners

`walk_batch` and `worker_function_batch` accept a `progress_callback(step, total_steps)` that fires every `progress_every` steps. This is what the bundled `_run_batch` uses to drive the `enlighten` bar; custom runners can use it to wire their own progress reporting without modifying the engine.

---

## 🚀 Getting Started

### Prerequisites

- Python **3.11** or **3.14**
- `pip` and a virtual environment tool

### Installation

```powershell
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install the package with development, notebook, and ML extras
pip install -e ".[dev,notebooks,ml]"
```

<details>
<summary>🧪 Running the test suite & coverage</summary>

```powershell
# Run all tests verbosely
pytest tests/ -v

# Run with coverage across all packages
pytest tests/ --cov=analysis --cov=domain --cov=engine --cov=persistence \
              --cov=utils --cov=visualization --cov=notebook_tools --cov-report=term
```

</details>

<details>
<summary>🎨 Linting & type checks</summary>

```powershell
ruff check engine/ domain/ analysis/ persistence/ utils/ visualization/ notebook_tools/ tests/
mypy .
```

</details>

---

## 🔄 Data Flow

```
data/*.json ──> persistence/loaders.py ──> ModelContext ──> Scenario ──> ParameterSet
                                                                            │
                                                                            ▼
notebooks/ <── ModelOutput <── worker_function(_batch) <── ParameterResolver
                                  ▲                ▲
                                  │                │
                       MarkovChains.walk   BatchMarkovChain.walk_batch
                                  ▲                ▲
                                  └────── TransitionGenerator ──────┘
                                              │
                                              ▼
                                  analysis/ (ICER, QALY, CEAC, EVPI)
                                              │
                                              ▼
                                  visualization/ (plots and figures)
```

`run_scenarios_in_batches` (in `notebook_tools/scenario_runner.py`) dispatches each scenario to one of two execution paths:

- **`engine="multiprocessing"` / `engine="pathos"`** — **scalar path.** Each `ModelInput` is shipped to a worker process and run through `MarkovChains.walk` + `worker_function`. Best when reward logic is custom or non-vectorizable.
- **`engine="batch"`** — **vectorized path.** The full input list for a scenario is processed in one call to `worker_function_batch`, which stacks the per-iter state into `(n_iters, n_states)` arrays and walks all iters in lockstep. ~13–17× faster for the standard hemophilia workload — see [⚡ Performance](#-performance).

---

## 🏛️ Design Principles

- **🔌 IO boundary separation** — All external data enters exclusively through `persistence/`.
- **🧼 Domain purity** — Clinical logic is isolated and independently testable in `domain/`.
- **🧩 Engine isolation** — Markov execution is fully decoupled from economic analysis.
- **🔁 Reproducibility-first PSA** — Every stochastic input is structured, typed, and seedable.
- **📒 Notebook-driven analysis** — Experiments live in notebooks, never in ad-hoc scripts.

---

## 📄 License

Released under the [MIT License](https://www.gnu.org/licenses/mit.html).

<p align="center">
  <sub>Built for transparent, reproducible health-economic research.</sub>
</p>
