TODO:

- report factor consumption per kg (as body weights calculates separately from steps, calculate it again on analysis file and add it to the pairs or maybe create a class for this pairs)
- asses quality adjusted life years per treatment arms, to validated reasonable effects on bleeding event on over all patient quality of life.
- test model accuracy if data comes available.

DONE:

- ICER plotted but it's intensive with matplotlib to draw
- outliers removed with cook's distance greater than 2 time
- arthropathy utility value implemented as continues decay per bleed event occurred
- passed number_of_bleeds from markov class to reward function for synchronizations
