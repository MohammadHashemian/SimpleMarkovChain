TODO:
- use datashader or simply use a subset of full data to avoid bottleneck processing issue with drawing scatter using matplotlib
- asses quality adjusted life years per treatment arms, to validated reasonable effects on bleeding event on over all patient quality of life.
- test model accuracy if data comes available.

DONE:
- ICER plotted but it's intensive with matplotlib to draw
- outliers removed with cook's distance greater than 2 time
- arthropathy utility value implemented as continues decay per bleed event occurred
- passed number_of_bleeds from markov class to reward function for synchronizations
