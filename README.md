<p align="center">
  <img src="https://raw.githubusercontent.com/opencode-ai/opencode/main/docs/logo.svg" width="0" alt="logo" />
</p>

<h1 align="center">🩸 Hemophilia Cost-Effectiveness Markov Model</h1>

<p align="center">
  A discrete-time Markov chain framework for health-economic modeling of hemophilia interventions, with full Probabilistic Sensitivity Analysis (PSA).
</p>

<p align="center">
  <a href="https://github.com/MohammadHashemian/SimpleMarkovChain/actions"><img src="https://github.com/MohammadHashemian/SimpleMarkovChain/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="#"><img src="https://img.shields.io/badge/tests-159%20passed-brightgreen" alt="tests" /></a>
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
- **🎯 Domain-rich rewards** — Pettersson score, bleeding events, factor consumption, utility weights, and mortality modifiers.
- **📊 Uncertainty quantification** — PSA (Monte Carlo), OWSA (tornado diagrams), CEAC, and EVPI.
- **🔬 Bayesian foundation** — PyMC-driven posterior sampling and convergence diagnostics (R-hat, ESS, divergences).
- **🧱 Typed IO boundary** — Pydantic schemas validate every external input before it touches the engine.
- **🧪 159 passing tests** spanning the engine, domain, analysis, and persistence layers.

---

## 📂 Directory Layout

| Component | Location | Responsibility |
|---|---|---|
| Input data (JSON) | `data/` | Clinical parameters, costs, utilities, mortality tables, simulation config |
| Pydantic schemas | `persistence/schemas/` | Typed input/output validation models |
| Data loaders | `persistence/loaders.py` | Reading and parsing JSON data sources |
| Model context | `persistence/context.py` | Aggregates all loaded data into a single `ModelContext` |
| Markov engine | `engine/` | Transition matrix construction, simulation runtime, runners |
| Domain logic | `domain/` | Health states, regime enums, scenarios, reward functions |
| Clinical rewards | `domain/rewards/` | Pettersson score, bleeding events, factor consumption, utility weights, mortality modifiers |
| PSA distributions | `analysis/psa/` | Sampling distributions, parameter uncertainty, Monte Carlo drivers |
| Economic analysis | `analysis/` | ICER, QALY aggregation, cost-effectiveness computations |
| General utilities | `utils/` | Pure helper functions (math, transformations, decorators, path utilities) |
| Visualization | `visualization/` | Plotting functions and result visualization utilities |
| Notebook tools | `notebook_tools/` | Helper modules shared across notebooks |
| Notebooks | `notebooks/` | Analysis notebooks (see below) |
| Tests | `tests/` | Pytest test suite |
| Outputs | `outputs/` | Generated figures, logs, and simulation results |

---

## 📓 Notebooks

| Notebook | Purpose |
|---|---|
| `00_poisson_mass_functions.ipynb` | Poisson mass function validation for bleeding event distributions |
| `01_preprocessing.ipynb` | Data preprocessing and parameter derivation |
| `02_meta_analysis.ipynb` | Bayesian meta-analysis of clinical inputs |
| `03_psa_simulation.ipynb` | Probabilistic sensitivity analysis (10,000 iterations) |
| `04_owsa_simulation.ipynb` | One-way sensitivity analysis simulation |
| `05_psa_analysis.ipynb` | PSA result analysis (CEAC, EVPI, scatter plots) |
| `07_owsa_analysis.ipynb` | OWSA result analysis (tornado diagrams) |

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
data/*.json  ──>  persistence/loaders.py  ──>  ModelContext  ──>  Scenario  ──>  ParameterSet
                                                                                       │
                                                                                       ▼
notebooks/  <──  MarkovResult  <──  MarkovChains  <──  TransitionGenerator  <──  ParameterResolver
                         │
                         ▼
               analysis/ (ICER, QALY, CEAC, EVPI)
                         │
                         ▼
               visualization/ (plots and figures)
```

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
