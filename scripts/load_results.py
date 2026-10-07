import json
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
SACRED = PROJECT / "epymarl" / "results" / "sacred"

TIME_KEY = "t"

# Columns identifying one training run. `alg` is the sacred results directory
# name (the yaml's `name:`, e.g. pac_sarsa_ns, which differs from the --config
# name pac_ns); `env` is the env key as stored, e.g.
# "lbf_envs:Foraging-5x5-3p-2f-budget-v0"; `run` is sacred's run
# number, the only thing separating two runs with the same seed.
RUN_KEYS = ["alg", "env", "seed", "run"]


def run_dir(alg, env, index, sacred=SACRED):
    """Path of one sacred run, e.g. ("pac_sarsa_ns", "matrix_envs:budget-table1-3p-v0", 1)."""
    path = Path(sacred) / alg / env / str(index)
    if not path.is_dir():
        raise FileNotFoundError(f"no sacred run at {path}")
    return path


def available_keys(alg, env, index, sacred=SACRED):
    """Sorted metric names logged by a run."""
    path = run_dir(alg, env, index, sacred) / "metrics.json"
    return sorted(json.loads(path.read_text()))


def load_run(alg, env, index, keys, sacred=SACRED, dropna=False):
    """DataFrame of the given metrics for one run, indexed by environment timestep.

    Returns columns [TIME_KEY, *keys] sorted by timestep. Keys logged on
    different step grids leave NaNs; pass dropna=True to keep only timesteps
    where every requested key has a value.
    """
    return _metrics_frame(run_dir(alg, env, index, sacred), keys, dropna=dropna)


def _metrics_frame(path, keys, dropna=False, skip_missing=False):
    """[TIME_KEY, *keys] for the run at `path`; see load_run."""
    if isinstance(keys, str):
        keys = [keys]

    metrics = json.loads((path / "metrics.json").read_text())

    missing = [k for k in keys if k not in metrics]
    if missing and not skip_missing:
        raise KeyError(f"{missing} not logged in {path}; available: {sorted(metrics)}")
    keys = [k for k in keys if k in metrics]

    frame = pd.DataFrame({TIME_KEY: []})
    for key in keys:
        series = pd.DataFrame({TIME_KEY: metrics[key]["steps"],
                               key: metrics[key]["values"]})
        series = series.drop_duplicates(subset=TIME_KEY, keep="last")
        frame = series if frame.empty else frame.merge(series, on=TIME_KEY, how="outer")

    frame = frame.sort_values(TIME_KEY).reset_index(drop=True)
    return frame.dropna(subset=keys) if dropna and keys else frame


def _config_value(config, key):
    """Look up a dotted config path, e.g. "env_args.time_limit". None if absent."""
    node = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def find_runs(algs=None, envs=None, seeds=None, sacred=SACRED, status="COMPLETED",
              config_keys=()):
    """One row per sacred run: [*RUN_KEYS, status, duration_s, *config_keys, path].

    algs/envs/seeds filter by exact value (a single value or a collection; None
    keeps everything); `status=None` keeps runs that failed or are still going.
    `config_keys` pulls dotted paths out of config.json -- e.g.
    config_keys=["q_nstep", "env_args.time_limit"] -- so runs can be grouped by
    the setting that actually differs rather than by seed number.
    """
    def wanted(value, allowed):
        if allowed is None:
            return True
        allowed = [allowed] if isinstance(allowed, (str, int)) else list(allowed)
        return value in allowed

    rows = []
    for alg_dir in sorted(Path(sacred).glob("*/")):
        if not wanted(alg_dir.name, algs):
            continue
        for env_dir in sorted(alg_dir.glob("*/")):
            if not wanted(env_dir.name, envs):
                continue
            for path in sorted(env_dir.iterdir(), key=lambda p: p.name):
                if not path.name.isdigit():
                    continue
                try:
                    config = json.loads((path / "config.json").read_text())
                    info = json.loads((path / "run.json").read_text())
                except (OSError, json.JSONDecodeError):
                    continue  # run died before writing its metadata
                if not wanted(config.get("seed"), seeds):
                    continue
                if status is not None and info.get("status") != status:
                    continue
                row = {"alg": alg_dir.name, "env": env_dir.name,
                       "seed": config.get("seed"), "run": int(path.name),
                       "status": info.get("status")}
                if info.get("start_time") and info.get("stop_time"):
                    row["duration_s"] = (datetime.fromisoformat(info["stop_time"])
                                         - datetime.fromisoformat(info["start_time"])
                                         ).total_seconds()
                row.update({k: _config_value(config, k) for k in config_keys})
                row["path"] = path
                rows.append(row)
    return pd.DataFrame(rows)


def load_runs(keys, algs=None, envs=None, seeds=None, sacred=SACRED, status="COMPLETED",
              config_keys=(), skip_missing=False, dropna=False):
    """Long frame of training curves: one row per (run, timestep).

    Columns are [*RUN_KEYS, TIME_KEY, *keys, *config_keys, duration_s]:

        df = load_runs(["test_return_mean"], config_keys=["q_nstep"])

    `skip_missing` tolerates runs that predate one of the keys (those rows get
    NaN instead of raising); `dropna` keeps only timesteps where every key has a
    value, which matters when keys are logged on different intervals.
    """
    runs = find_runs(algs, envs, seeds, sacred, status, config_keys)
    if runs.empty:
        return runs

    frames = []
    for row in runs.to_dict("records"):
        path = row.pop("path")
        frame = _metrics_frame(path, keys, dropna=dropna, skip_missing=skip_missing)
        frames.append(frame.assign(**row))

    out = pd.concat(frames, ignore_index=True)
    # leading columns in a predictable order; keys no run logged are simply absent
    columns = [c for c in [*RUN_KEYS, TIME_KEY, *([keys] if isinstance(keys, str) else keys)]
               if c in out.columns]
    out = out[columns + [c for c in out.columns if c not in columns]]
    return out.sort_values([*RUN_KEYS, TIME_KEY], ignore_index=True)


def final_values(runs, keys=None, by=("alg", "env")):
    """Last logged value of each key per run, then mean/std over the `by` groups.

    `runs` is a long frame from load_runs. Returns a frame indexed by `by` with
    (key, "mean"/"std"/"runs") columns -- the usual end-of-training table.
    """
    keys = list(keys or [c for c in runs.columns
                         if c not in (*RUN_KEYS, TIME_KEY, "status", "duration_s")
                         and pd.api.types.is_numeric_dtype(runs[c])])
    run_cols = [c for c in RUN_KEYS if c in runs.columns]
    # last logged value of each metric, but the *run's* value of everything else
    # (config columns are constant within a run, so "first" just carries them)
    how = {k: "last" for k in keys}
    how.update({c: "first" for c in runs.columns
                if c not in (*keys, TIME_KEY, *run_cols)})
    last = runs.sort_values(TIME_KEY).groupby(run_cols, as_index=False).agg(how)
    grouped = last.groupby(list(by))[keys]
    summary = grouped.agg(["mean", "std"])
    summary[("runs", "")] = grouped.size()
    return summary
