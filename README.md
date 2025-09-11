TODO:
- Mortality rate Disease and natural related cause (All Cause, Disease mortality) [Partially]
- What is the quality of patients life with lower versus higher abrs, is that linear or exponential?
- What is the optimal ABR value to prefer prophylaxis as cost-effective treatment?
- Test model accuracy if data comes available.
- Simulate cost-effectiveness of late prophylaxis treatment for arthropathy patients as they gain less qaly from prophylaxis treatment

DONE:
- Bias with prophylaxis vs on demand for ABR report [Checked]
- Age of starting prophylaxis [Checked]
- Advantage and disadvantage of different Markov models, Cycles length. [Checked]
- Remove discount rate [Checked]
- Asses quality adjusted life years per treatment arms [Checked]
- Report factor consumption per kg [Checked]
- ICER plotted but it's intensive with matplotlib to draw # RESOLVED
- outliers removed with cook's distance greater than 2 time # DEPRECATED
- arthropathy utility value implemented as continues decay per bleed event occurred # DEPRECATED
- passed number_of_bleeds from markov class to reward function for synchronizations

Questions:
- Bootstrapping or individual point icer calculation?
- Does the model need to add Pettersson score for arthropathy utility progression?
- What does it mean the incremental costs . Incremental QALYS plot? where the values come from? it that a bootstrapping?
- What values and parameters need to be reported? should i report means (SD) or median (IQR)?
- Which model inputs need to be associated for uncertainty?
- Is it right to sample ABR from different population of same article reports? [CHECKED]
- Should the natural mortality rate of population be added to population simulation to calculated the burden of disease? [CHECKED]