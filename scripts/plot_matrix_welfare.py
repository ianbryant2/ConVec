"""Summed env return of the three agents over gradient updates for PAC, QMIX,
MAPPO and VDN on the matrix_welfare sweep (Table 1, budgets 20/20/20): the
mean over seeds shaded by one standard error, and the social-welfare optimum
(B, A, B) = 32 as a dotted line. Same style as plot_matrix_table1.py.

    python scripts/plot_matrix_welfare.py  ->  figures/matrix_welfare_returns.pdf
"""
from plot_matrix_table1 import FIGURES, RETURN, load, plot

OPTIMUM = 32.0

runs = load("matrix_welfare", [RETURN])
plot(runs, RETURN, "Summed return of the 3 agents", FIGURES / "matrix_welfare_returns.pdf",
     ylim=(None, OPTIMUM + 2), ref=(OPTIMUM, "social-welfare optimum, (B, A, B) = 32"),
     ref_above=True)
