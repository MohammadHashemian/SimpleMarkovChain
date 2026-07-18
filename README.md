# Project Structure

This project follows a layered architecture designed for reproducible health-economic Markov simulations with PSA (Probabilistic Sensitivity Analysis).

Data ingestion happens at the IO boundary layer, while simulation, domain logic, and analysis are strictly separated to ensure modularity and reproducibility.

---

## 📁 Directory Layout

| Component                                | Location                 | Responsibility                                                                    |
| ---------------------------------------- | ------------------------ | --------------------------------------------------------------------------------- |
| JSON data files                          | `data/`                  | Raw input data (costs, currencies, parameters, etc.)                              |
| Pydantic schemas                         | `persistence/schemas.py` | Typed input/output validation models                                              |
| Data loaders                             | `persistence/loaders.py` | Reading and parsing external data sources                                         |
| Path / logging utilities                 | `persistence/`           | File system utilities, logging, experiment IO                                     |
| Markov engine (core simulation runtime)  | `engine/`                | Simulation execution, transition matrix construction, runner logic                |
| Domain reward functions                  | `domain/rewards/`        | Clinical logic (utilities, bleeding models, factor consumption, Pettersson score) |
| PSA (Probabilistic Sensitivity Analysis) | `analysis/psa/`          | Sampling distributions, parameter uncertainty, Monte Carlo drivers                |
| Economic analysis                        | `analysis/`              | ICER, QALY aggregation, cost-effectiveness computations                           |
| Visualization                            | `viz/`                   | Plotting functions and result visualization utilities                             |
| General utilities                        | `utils/`                 | Pure helper functions (math, transformations, decorators)                         |
| Notebooks                                | `notebooks/`             | Exploratory analysis and experiments                                              |
| Outputs                                  | `outputs/`               | Generated figures, logs, and simulation results                                   |

---

## 🧠 Design Principles

- **IO boundary separation**: All external data enters through `persistence/`
- **Domain purity**: Clinical logic is isolated in `domain/`
- **Engine isolation**: Markov execution is fully independent of analysis
- **Reproducibility-first PSA design**: All stochastic inputs are structured and typed
- **Separation of concerns**: Cost, utility, and simulation logic are decoupled

---

## Data Flow

TO FIT:
[TransitionGenerator] - [WorkerFunction] - [RewardFunctions]

Raw Data -> Loaders -> ModelContext -> ScenarioBuilder -> ParameterTemplate -> PSASampler -> ParameterSet -> Engine
