# Project Status & Roadmap

This document outlines the current status, completed work, and future plans for the cost-effectiveness model.

## 1. Planned Future Work (TODO)

### Model Validation
- Conduct formal model validation if external data becomes available to test predictive accuracy.

### Policy Analysis
- Simulate the cost-effectiveness of initiating prophylaxis later in life for patients with existing arthropathy, as the potential quality-adjusted life year (QALY) gain may be different.

## 2. Recently Completed (DONE)

The core model structure and key analyses have been implemented and finalized.

### Model Framework & Parameters
- Verified handling of bias between prophylaxis and on-demand treatment for the Annual Bleeding Rate (ABR) input
- Incorporated the age at which prophylaxis is started as a key variable
- Evaluated and selected appropriate Markov model structures and cycle lengths
- Removed discounting from the analysis as per the project scope

### Key Outputs Calculated
- Implemented calculation of Quality-Adjusted Life Years (QALYs) for each treatment arm
- Reported factor consumption normalized per kilogram of patient weight
- Resolved technical issues with plotting the Incremental Cost-Effectiveness Ratio (ICER)

### Clinical Progression
- Synchronized the Markov model states with the bleed counting function
- Merged bleed and hemarthrosis tracking into a single, efficient function to ensure accuracy
- Implemented all-cause mortality, accounting for both background (natural) and disease-related rates

### Analysis & Insights
- Investigated the relationship between ABR and quality of life (determined to be non-linear)
- Identified the threshold ABR value at which prophylaxis becomes a cost-effective treatment option

## 3. Open Questions & Decisions Needed

### Methodological Decisions
- **Bootstrapping vs. Point Estimates:** Should we use bootstrapping for uncertainty analysis or calculate individual point ICERs?
- **Arthropathy Modeling:** Does the model need to incorporate Pettersson score for more accurate utility progression?
- **Uncertainty Parameters:** Which specific model inputs should be associated with probabilistic uncertainty analysis?

### Results Interpretation
- **Incremental Cost-Effectiveness Plot:** What is the proper interpretation of the incremental costs vs. incremental QALYs plot, and where do the values originate?
- **Results Reporting:** Should results be reported as means (SD) or medians (IQR)?

### Data Sourcing
- **Parameter Sampling:** Is it methodologically sound to sample ABR values from different populations within the same study?
- **Mortality Rates:** Was it correct to add natural mortality rates to the population simulation to calculate the disease burden?