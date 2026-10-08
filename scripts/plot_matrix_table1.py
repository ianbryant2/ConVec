"""PAC, QMIX, MAPPO and VDN on the matrix_table1 sweep (Table 1, budgets
(6, 6, 5)) over gradient updates: the mean over seeds shaded by one standard
error.

  - figures/matrix_table1_budget.pdf: fraction of greedy test episodes whose
    joint action is within budget under the true payoffs (test_feasible_mean).
  - figures/matrix_table1_returns.pdf: summed env return of the three agents,
    with the best within-budget welfare, (A, A, A) = 30.

    python scripts/plot_matrix_table1.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweeps import engine  # noqa: E402

FEASIBLE, RETURN = "test_feasible_mean", "test_env_return_total_mean"
# Parallel-runner algs make one update per 10 one-step episodes, episode-runner
# algs one per episode (algs.yaml step_scale).
STEPS_PER_UPDATE = {"pac_ns": 10, "mappo_ns": 10, "qmix_ns": 1, "vdn_ns": 1}
# Legend order, colours and line styles of the paper figure; the styles keep
# MAPPO and VDN apart for colour-blind readers.
ALGS = {
    "pac_ns": ("PAC", "#3a76d8", "-"),
    "qmix_ns": ("QMIX", "#a64fc9", "--"),
    "mappo_ns": ("MAPPO", "#e08540", "-."),
    "vdn_ns": ("VDN", "#4aa564", ":"),
}
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def load(sweep, keys, **where):
    """The sweep's curves with an `updates` column (gradient updates)."""
    runs = engine.load_runs(sweep, keys, dropna=True, **where)
    runs["updates"] = runs.t / runs.alg.map(STEPS_PER_UPDATE)
    return runs


def plot(runs, key, ylabel, out, ylim=None, ref=None, ref_above=False):
    """Mean over seeds per alg, shaded by one standard error; `ref` is a
    (value, label) drawn as a dotted horizontal line, labelled below it unless
    ref_above."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for alg, (name, color, ls) in ALGS.items():
        stats = runs[runs.alg == alg].groupby("updates")[key].agg(["mean", "sem"])
        x, mean, sem = stats.index, stats["mean"], stats["sem"].fillna(0)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.15, lw=0)
        ax.plot(x, mean, color=color, ls=ls, lw=2, label=name)
    if ref is not None:
        value, text = ref
        ax.axhline(value, color="#52514e", lw=1, ls=(0, (1, 2)))
        ax.annotate(text, (runs.updates.max() * 0.98, value), ha="right",
                    va="bottom" if ref_above else "top", xytext=(0, 3 if ref_above else -3),
                    textcoords="offset points", fontsize=8, color="#52514e")
    ax.set_xlim(0, runs.updates.max())
    if ylim:
        ax.set_ylim(*ylim)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f}k")
    ax.set_xlabel("Gradient updates")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False,
              title=f"mean ± 1 s.e., {runs.seed.nunique()} seeds", title_fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(out)


if __name__ == "__main__":
    runs = load("matrix_table1", [FEASIBLE, RETURN], steps="50k")
    plot(runs, FEASIBLE, "Expected budget satisfied", FIGURES / "matrix_table1_budget.pdf",
         ylim=(-0.02, 1.02))
    plot(runs, RETURN, "Summed return of the 3 agents", FIGURES / "matrix_table1_returns.pdf",
         ref=(30.0, "best within budget, (A, A, A) = 30"))
