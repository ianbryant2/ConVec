"""Standard result plots.
Input is always a *long* frame -- one row per (run, timestep) -- as produced by
`load_results.load_runs()`:

    alg   env                     seed  t       test_return_mean  q_nstep
    pac_. lbf_envs:Foraging-5x5-. 0     140000  0.11              10
    pac_. lbf_envs:Foraging-5x5-. 0     280000  0.19              10

Typical use -- one panel per environment, algorithms as series:

    import plotting as pl

    fig, axes = pl.grid(len(envs), ncols=3)
    for ax, env in zip(axes, envs):
        pl.panel(df[df["env"] == env], y="test_return_mean", hue="alg",
                 order=ALGS, ax=ax, title=env, ylabel="greedy test return")
    pl.legend(fig)
    pl.finish(fig, suptitle="...", note="shading = min/max over seeds")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# Categorical slots, taken in order and never cycled: a series keeps its slot no
# matter which other series share the panel. Validated for colour-vision
# deficiency as an ordered set (worst adjacent-pair CVD dE 9.1, normal-vision
# 19.6); past 8 series, pass explicit `colors` rather than generating a 9th hue.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

TEXT, MUTED, GRID, SPINE = "#333333", "#767676", "#dddddd", "#bbbbbb"

BAND_ALPHA = 0.15
DASHED, DOTTED = (0, (4, 4)), (0, (1, 2.5))

# Columns that identify one training run, used to align/trim curves. Any subset
# that is actually present in the frame is used.
RUN_KEYS = ("alg", "env", "seed", "run")

TIME_KEY = "t"      # x column written by load_results
METRIC_KEY = "metric"  # hue column synthesised when `y` is a list of metrics


def assign_colors(keys, colors=None):
    """{key: hex} taking PALETTE slots in order; `colors` pins specific keys.

    Pass the resulting dict (or the same `order`) to every panel in a figure so
    that a series keeps its colour even in panels where other series are absent.
    """
    assigned = dict(colors or {})
    free = [c for c in PALETTE if c not in assigned.values()]
    for key in keys:
        if key in assigned:
            continue
        if not free:
            raise ValueError(
                f"more than {len(PALETTE)} series ({key!r} is unassigned): split "
                "the panel or pass explicit colors={...} rather than cycling hues")
        assigned[key] = free.pop(0)
    return assigned


def summarise(data, y, x=TIME_KEY, hue=None, agg="mean", band="minmax", trim=True,
              align="auto"):
    """Collapse the runs in a long frame into one curve (+ band) per series.

    Returns a frame with columns [hue?, x, "value", "lo", "hi"], one row per
    (series, timestep). `agg` is "mean" or "median"; `band` is "minmax", "std",
    "sem", "iqr" or None (lo == hi == value).

    Seeds rarely evaluate at *exactly* the same environment step (sacred logs
    the step count reached, not the requested interval), which would leave every
    x value with a single run on it. So with align="auto" curves whose x grids
    disagree are matched on evaluation index instead, and plotted against the
    mean step count of that evaluation; align="x" forces literal x matching.

    With `trim`, every series is cut at the last evaluation *all* of its runs
    reached, so a short run cannot bend the tail of the average.
    """
    frame = data.dropna(subset=[y]).sort_values(x)
    run_cols = [c for c in RUN_KEYS if c in frame.columns]
    groups = [(None, frame)] if hue is None else frame.groupby(hue, sort=False)

    out = []
    for key, group in groups:
        if run_cols:
            runs = group.groupby(run_cols)
            by_index = align == "index" or (align == "auto"
                                            and group[x].nunique() > runs.size().max())
            group = group.assign(_point=runs.cumcount() if by_index else group[x])
            if trim:
                group = group[group["_point"] <= group.groupby(run_cols)["_point"].max().min()]
        else:
            group = group.assign(_point=group[x])

        points = group.groupby("_point")
        stat = points[y].agg(["mean", "median", "std", "min", "max", "count"])
        value = stat[agg]
        if band == "iqr":
            spread = (points[y].quantile(0.25), points[y].quantile(0.75))
        else:
            spread = {
                None: (value, value),
                "minmax": (stat["min"], stat["max"]),
                "std": (value - stat["std"], value + stat["std"]),
                "sem": (value - stat["std"] / np.sqrt(stat["count"]),
                        value + stat["std"] / np.sqrt(stat["count"])),
            }[band]
        curve = pd.DataFrame({x: points[x].mean(), "value": value,
                              "lo": spread[0], "hi": spread[1]}).reset_index(drop=True)
        if hue is not None:
            curve.insert(0, hue, key)
        out.append(curve)
    return pd.concat(out, ignore_index=True)


def panel(data, y, x=TIME_KEY, hue=None, ax=None, *, order=None, colors=None,
          labels=None, agg="mean", band="minmax", trim=True, align="auto",
          title=None, xlabel=None, ylabel=None, legend=False, lw=2,
          figsize=(4.4, 3.6)):
    """Draw one metric against training time onto `ax` (created if None).

    data    long frame, one row per (run, timestep)
    y       metric column, or a list of columns to draw as series against each
            other (only ever put metrics that share a scale on one axis)
    hue     column whose values become the series (e.g. "alg", "q_nstep")
    order   series to draw, in colour-slot order; also filters the data. Pass
            the same `order` (or `colors`) to every panel of a figure so a
            series keeps its colour across panels. Series with no rows here are
            silently skipped, so one `order` works for a ragged grid.
    labels  {series value: display label} for the legend
    colors  {series value: hex} pinning specific series

    Returns the Axes, so the caller can keep tweaking it.
    """
    if isinstance(y, (list, tuple)):
        if hue is not None:
            raise ValueError("pass either a list of `y` metrics or a `hue`, not both")
        id_cols = [c for c in (*RUN_KEYS, x) if c in data.columns]
        data = data.melt(id_vars=id_cols, value_vars=list(y),
                         var_name=METRIC_KEY, value_name="value")
        order, hue, y = order or list(y), METRIC_KEY, "value"

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if hue is None:
        series, colors = [None], {None: colors} if isinstance(colors, str) else colors
    else:
        series = list(order) if order is not None else sorted(data[hue].unique())
        data = data[data[hue].isin(series)]
    colors = assign_colors(series, colors)
    labels = labels or {}

    if not data.empty:
        curves = summarise(data, y, x=x, hue=hue, agg=agg, band=band, trim=trim,
                           align=align)
        for key in series:
            curve = curves if hue is None else curves[curves[hue] == key]
            if curve.empty:
                continue
            if band is not None:
                ax.fill_between(curve[x], curve["lo"], curve["hi"],
                                color=colors[key], alpha=BAND_ALPHA, lw=0)
            ax.plot(curve[x], curve["value"], color=colors[key], lw=lw,
                    label=str(labels.get(key, key)))

    ax.set_title(title, loc="left", fontsize=9.5, color=TEXT, pad=8)
    ax.set_xlabel(xlabel if xlabel is not None else "environment steps",
                  fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel if ylabel is not None else y, fontsize=9, color=MUTED)
    style_axis(ax)
    if legend and hue is not None:
        # below the panel, never inside it: no placement can overlap a curve
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False,
                  ncol=min(len(series), 3), fontsize=8, labelcolor=TEXT,
                  handlelength=1.5, columnspacing=1.2)
    return ax


def reference_line(ax, value, style=DASHED, label=None):
    """Horizontal reference (optimum, random/scripted baseline, zero, ...)."""
    ax.axhline(value, color=MUTED, lw=1, ls=style, zorder=1,
               label=label if label else None)
    return ax


def grid(n, ncols=3, width=4.4, height=3.6, sharey=False):
    """(fig, axes) with exactly `n` visible panels laid out `ncols` wide.

    Returns axes as a flat array of length n. Slots left over in the last row
    stay hidden, and `legend()` reuses the first of them as a legend slot rather
    than floating the legend over a panel.
    """
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, sharey=sharey,
                             figsize=(width * ncols, height * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    if sharey:
        for ax in axes[:n]:
            ax.label_outer()
    return fig, axes[:n]


def legend(fig, order=None, labels=None, colors=None, ncol=None):
    """One legend for the whole figure, built from every panel's series.

    Uses a spare (hidden) axes from `grid()` if there is one, otherwise puts the
    legend below the figure. Pass `order` (+ `labels`/`colors`, same as `panel`)
    to list series that appear in only some panels -- or in none of them.
    """
    if order is not None:
        colors, labels = assign_colors(order, colors), labels or {}
        handles = [Line2D([0], [0], color=colors[key], lw=2,
                          label=str(labels.get(key, key))) for key in order]
    else:  # collect from the panels, first appearance wins, de-duplicated by label
        seen = {}
        for ax in fig.axes:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(label, handle)
        handles = list(seen.values())
    texts = [handle.get_label() for handle in handles]

    spares = [ax for ax in fig.axes if not ax.get_visible()]
    if spares:
        slot = spares[0]
        slot.set_visible(True)
        slot.axis("off")
        return slot.legend(handles, texts, loc="center left", fontsize=8.5,
                           frameon=False, labelcolor=TEXT, handlelength=1.5)
    return fig.legend(handles, texts, loc="lower center", bbox_to_anchor=(0.5, -0.04),
                      ncol=ncol or min(len(handles), 4), frameon=False, fontsize=8,
                      labelcolor=TEXT, handlelength=1.5, columnspacing=1.2)


def finish(fig, suptitle=None, note=None, rect=(0, 0, 1, 0.96)):
    """Figure title (left) + caption (right), then tight_layout.

    Returns nothing on purpose: as the last statement of a notebook cell,
    returning the figure would render it a second time.
    """
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=12, color=TEXT)
    if note:
        fig.text(0.99, 0.995, note, ha="right", fontsize=8, color=MUTED)
    fig.tight_layout(rect=list(rect))


def style_axis(ax, si_x=True):
    """Recessive grid and axes; SI-suffixed x ticks (140k, 1.4M) for step counts."""
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    if si_x:
        ax.xaxis.set_major_formatter(FuncFormatter(si_format))
    return ax


def si_format(value, _pos=None):
    """1_400_000 -> '1.4M'; small numbers are left alone."""
    for scale, suffix in ((1e6, "M"), (1e3, "k")):
        if abs(value) >= scale:
            return f"{value / scale:g}{suffix}"
    return f"{value:g}"
