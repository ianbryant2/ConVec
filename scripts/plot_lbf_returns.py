"""Each agent's discounted test return over training, for the lbf_td_converge_a{0,1,2}
sweeps (LBF, one agent on budget 0.55 and the others on 0.8): the mean over seeds
shaded by one standard error, the step the critic is frozen, and each agent's
break-even return. One row per sweep, titled with its concession vector c.

    python scripts/plot_lbf_returns.py  ->  figures/lbf_disc_returns.pdf
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweeps import engine  # noqa: E402

KEYS = [f"test_env_disc_return_{i}_mean" for i in range(3)]
EVAL_EVERY = 50_000     # test interval; seeds log at slightly different t
SMOOTH = 3              # centred rolling window, in evaluations
CRITIC_FROZEN = 500_000
# Break-even discounted return E[V*_i(s_0)] - c_i: E[V*_i(s_0)] = 0.978 over 2000
# random layouts (main's scripts/lbf_feasibility.py exact solver, gamma 0.99).
V_STAR = 0.978
# sweep -> (concession vector c, series colour)
SWEEPS = {
    "lbf_td_converge_a0": ((0.55, 0.8, 0.8), "#2a78d6"),
    "lbf_td_converge_a1": ((0.8, 0.55, 0.8), "#d4622a"),
    "lbf_td_converge_a2": ((0.8, 0.8, 0.55), "#2a9a63"),
}
FIGURES = Path(__file__).resolve().parent.parent / "figures"

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "pdf.fonttype": 42,  # embed TrueType so the text stays selectable
})


def plot_row(subfig, sweep, budget, series, bottom_row):
    runs = engine.load_runs(sweep, KEYS)
    runs["t"] = (runs.t / EVAL_EVERY).round().astype(int) * EVAL_EVERY

    axes = subfig.subplots(1, 3, sharex=True)
    for i, ax in enumerate(axes):
        key, target = KEYS[i], V_STAR - budget[i]
        stats = runs.groupby("t")[key].agg(["mean", "sem"]).sort_index()
        stats = stats.rolling(SMOOTH, min_periods=1, center=True).mean()
        lo, hi = stats["mean"] - stats["sem"], stats["mean"] + stats["sem"]
        ax.fill_between(stats.index, lo, hi, color=series, alpha=0.2, lw=0)
        ax.plot(stats.index, stats["mean"], color=series, lw=1.2)
        ax.axvline(CRITIC_FROZEN, color=MUTED, ls="--", lw=0.8)
        ax.axhline(target, color=INK, ls=":", lw=1.1)
        ax.text(CRITIC_FROZEN, target, f" break-even {target:.3f}", ha="left", va="top",
                fontsize=6.5, color=INK)
        ax.set_title(f"Agent {i} ($c_{i}$ = {budget[i]:g})", loc="left", color=INK, pad=3)
        ax.grid(axis="y", color=GRID, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
            ax.spines[s].set_linewidth(0.6)
        ax.tick_params(colors=MUTED, width=0.6, length=2.5)
        bottom, top = min(lo.min(), target), max(hi.max(), target)
        pad = 0.08 * (top - bottom)
        ax.set_ylim(bottom - pad, top + pad)
        if bottom_row:
            ax.set_xlabel("Environment steps", color=MUTED)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1e6:g}M" if x else "0")
        ax.set_xticks(range(0, 7_000_001, 1_000_000))

    axes[0].set_ylabel("Discounted return", color=MUTED)
    axes[0].text(CRITIC_FROZEN, axes[0].get_ylim()[1], " critic frozen", fontsize=6.5, color=MUTED, va="top")
    c = ", ".join(f"{b:g}" for b in budget)
    subfig.suptitle(f"Concession budgets c = ({c})", x=0.01, ha="left", color=INK, fontsize=9)
    return runs.seed.nunique()


FIGURES.mkdir(exist_ok=True)
fig = plt.figure(figsize=(6.75, 6.6), layout="constrained")
for row, (subfig, (sweep, (budget, series))) in enumerate(zip(fig.subfigures(len(SWEEPS), 1), SWEEPS.items())):
    seeds = plot_row(subfig, sweep, budget, series, bottom_row=row == len(SWEEPS) - 1)
    print(f"{sweep}: mean ± 1 s.e. over {seeds} seeds")
out = FIGURES / "lbf_disc_returns.pdf"
fig.savefig(out)
fig.savefig(out.with_suffix(".png"), dpi=300, facecolor="white")
print(out)
