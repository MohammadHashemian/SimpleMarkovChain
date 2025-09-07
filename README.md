TODO:
- what is the quality of patients life with lower versus higher abrs, is that linear or exponential?
- what is the optimal ABR value to prefer prophylaxis as cost-effective treatment?
- test model accuracy if data comes available.
- simulate cost-effectiveness of late prophylaxis treatment for arthropathy patients as they gain less qaly from prophylaxis treatment

DONE:
- Asses quality adjusted life years per treatment arms
- Report factor consumption per kg
- ICER plotted but it's intensive with matplotlib to draw
- outliers removed with cook's distance greater than 2 time
- arthropathy utility value implemented as continues decay per bleed event occurred
- passed number_of_bleeds from markov class to reward function for synchronizations
