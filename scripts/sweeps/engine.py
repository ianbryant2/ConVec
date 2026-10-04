"""YAML sweep engine: crosses a sweep's axes, resolves each run's settings,
launches epymarl and tracks which runs are done.

A sweep (scripts/sweeps/<name>.yaml) lists seeds, base settings, axes to
cross (alg, env, then any others) and optional combinations to exclude.
Each run's settings merge presets defaults, the sweep base, the alg/env
presets and each axis option, later winning. t_max and test_interval are
then multiplied by algs[alg].step_scale[env family]. This is useful when algorithims 
use the parallel runner.

Runs are tagged in sacred's --comment with a hash of their settings; a run
counts as done when a COMPLETED run has the same hash and seed. It is stale
when a COMPLETED run has the same sweep, axis options and seed but different
settings.
"""
import argparse
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
PRESETS = ROOT / "presets"
PROJECT = ROOT.parent.parent
EPYMARL = PROJECT / "epymarl"
SACRED = EPYMARL / "results" / "sacred"
LOGS = EPYMARL / "results" / "sweep_logs"

STEP_KEYS = ["t_max", "test_interval"]
LOG_INTERVAL_KEYS = ["log_interval", "runner_log_interval", "learner_log_interval"]
FIXED_AXES = ("alg", "env")
# A RUNNING run whose heartbeat is older than this died without saying so.
STALE_HEARTBEAT = timedelta(minutes=15)
# `run` asks for --yes before launching more than this many runs.
CONFIRM_ABOVE = 50


# --- loading -----------------------------------------------------------------

def _yaml(path):
    return yaml.safe_load(Path(path).read_text()) or {}


def load_presets(directory=PRESETS):
    """Merge presets/*.yaml into {defaults, envs, algs, profiles}. Top-level
    keys starting with "_" only hold YAML anchors and are ignored."""
    presets = {"defaults": {}, "envs": {}, "algs": {}, "profiles": {}}
    for path in sorted(Path(directory).glob("*.yaml")):
        for section, entries in _yaml(path).items():
            if section.startswith("_"):
                continue
            if section not in presets:
                raise ValueError(f"{path.name}: unknown section '{section}' "
                                 f"(expected {sorted(presets)})")
            clash = set(presets[section]) & set(entries)
            if clash:
                raise ValueError(f"{path.name}: {section} {sorted(clash)} defined twice")
            presets[section].update(entries)
    return presets


def sweep_names():
    return sorted(p.stem for p in ROOT.glob("*.yaml"))


@dataclass
class Sweep:
    name: str
    seeds: list
    base: object
    axes: dict  # axis -> {option: spec}; alg/env options map to their own name
    exclude: list = field(default_factory=list)
    presets: dict = field(default_factory=load_presets)


def load_sweep(name):
    path = ROOT / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"no sweep '{name}' (available: {sweep_names()})")
    raw = _yaml(path)
    unknown = set(raw) - {"seeds", "base", "axes", "exclude"}
    if unknown:
        raise ValueError(f"{path.name}: unknown keys {sorted(unknown)}")
    axes = {}
    for axis, options in raw["axes"].items():
        if axis in FIXED_AXES:
            if not isinstance(options, list):
                raise ValueError(f"{path.name}: axis '{axis}' must be a list of names")
            axes[axis] = {o: o for o in options}
        elif isinstance(options, dict):
            axes[axis] = options
        else:
            raise ValueError(f"{path.name}: axis '{axis}' must map option names to settings")
    missing = [a for a in FIXED_AXES if a not in axes]
    if missing:
        raise ValueError(f"{path.name}: needs axes {missing}")
    return Sweep(name, list(raw.get("seeds", [0])), raw.get("base"), axes,
                 raw.get("exclude") or [])


# --- resolving one run's settings ------------------------------------------

def flatten(d, prefix=""):
    """Nested settings -> {dotted key: value}. Empty dicts are kept as values."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _spec_layers(spec, profiles, source, seen=()):
    """A spec -> [(source, flat settings)], profiles expanded in place."""
    if spec is None:
        return []
    if isinstance(spec, str):
        if spec not in profiles:
            raise ValueError(f"{source}: unknown profile '{spec}' (available: {sorted(profiles)})")
        if spec in seen:
            raise ValueError(f"profile cycle: {' -> '.join(seen + (spec,))}")
        return _spec_layers(profiles[spec], profiles, f"profile:{spec}", seen + (spec,))
    if isinstance(spec, list):
        return [layer for item in spec for layer in _spec_layers(item, profiles, source, seen)]
    if isinstance(spec, dict):
        return [(source, flatten(spec))]
    raise ValueError(f"{source}: settings must be a dict, profile name or list, got {spec!r}")


def _env_preset(presets, name, seen=()):
    """(family, [(source, flat settings)]) of an env, following `extends`."""
    envs = presets["envs"]
    if name not in envs:
        raise ValueError(f"unknown env '{name}' (available: {sorted(envs)})")
    if name in seen:
        raise ValueError(f"env extends cycle: {' -> '.join(seen + (name,))}")
    env = envs[name]
    family, layers = None, []
    if "extends" in env:
        family, layers = _env_preset(presets, env["extends"], seen + (name,))
    family = env.get("family", family)
    if family is None:
        raise ValueError(f"env '{name}' has no family")
    return family, layers + [(f"envs.{name}", flatten(env.get("settings") or {}))]


def _alg_runner(alg):
    """The runner the alg's yaml uses (default.yaml's "episode" if it sets none)."""
    text = (EPYMARL / "src" / "config" / "algs" / f"{alg}.yaml").read_text()
    match = re.search(r'^runner:\s*"?([^"\s#]+)"?', text, re.M)
    return match.group(1) if match else "episode"


def _default_budget_rule():
    config = _yaml(EPYMARL / "src" / "config" / "envs" / "gymma_budget.yaml")
    return config["budget"]["rule"]


def _rule_is_common(rule):
    """Whether a budget rule pays a common reward, read from rules.py without
    importing epymarl's env packages."""
    import importlib.util
    path = EPYMARL / "src" / "envs" / "budget" / "rules.py"
    spec = importlib.util.spec_from_file_location("budget_rules", path)
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    return rules.get_rule(rule).common


def _normalise(v):
    """Values compared/hashed the same however YAML or Python wrote them."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, (list, tuple)):
        return [_normalise(x) for x in v]
    if isinstance(v, dict):
        return {k: _normalise(x) for k, x in v.items()}
    return v


def settings_hash(settings, launch):
    body = {"settings": {k: v for k, v in settings.items() if k != "seed"}, "launch": launch}
    text = json.dumps(_normalise(body), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass
class Run:
    sweep: str
    axes: dict      # axis -> option, in axis order
    seed: int
    settings: dict  # flat sacred overrides, seed included
    sources: dict   # key -> where its value came from
    launch: dict    # {"env_config", "args"}: --env-config and the extra sacred args
    hash: str
    overrides: dict = field(default_factory=dict)  # --set, part of the run's identity

    @property
    def alg(self):
        return self.axes["alg"]

    @property
    def label(self):
        return ",".join(f"{k}={v}" for k, v in self.axes.items()) + f",seed={self.seed}"

    @property
    def tag(self):
        tag = {"sweep": self.sweep, "axes": self.axes, "seed": self.seed, "hash": self.hash}
        if self.overrides:
            tag["set"] = self.overrides
        return tag

    @property
    def identity(self):
        return _identity(self.tag)

    def command(self):
        args = {**self.settings, **self.launch["args"]}
        return [sys.executable, "src/main.py", f"--config={self.alg}",
                f"--env-config={self.launch['env_config']}",
                "--comment=" + json.dumps(self.tag, separators=(",", ":")),
                "with"] + [f"{k}={v}" for k, v in args.items()]


def resolve(sweep, axes, seed, overrides=None):
    """The Run for one combination of axis options and a seed. overrides
    ({dotted key: value}, from --set) are applied last, as literal final
    values, and so change the run's hash."""
    presets = sweep.presets
    alg_name, env_name = axes["alg"], axes["env"]
    if alg_name not in presets["algs"]:
        raise ValueError(f"unknown alg '{alg_name}' (available: {sorted(presets['algs'])})")
    alg = presets["algs"][alg_name]
    family, env_layers = _env_preset(presets, env_name)
    if family not in alg:
        raise ValueError(f"alg '{alg_name}' has no settings for env family '{family}'")

    layers = [("defaults", flatten(presets["defaults"]))]
    layers += _spec_layers(sweep.base, presets["profiles"], "base")
    layers += [(f"algs.{alg_name}.{family}", flatten(alg[family] or {}))]
    layers += env_layers
    for axis, option in axes.items():
        if axis not in FIXED_AXES:
            layers += _spec_layers(sweep.axes[axis][option], presets["profiles"],
                                   f"{axis}={option}")

    settings, sources = {}, {}
    for source, flat in layers:
        for k, v in flat.items():
            settings[k] = v
            sources[k] = source

    scale = (alg.get("step_scale") or {}).get(family, 1)
    if scale != 1:
        for k in STEP_KEYS:
            if k in settings:
                settings[k] *= scale
                sources[k] += f" x{scale} (algs.{alg_name}.step_scale)"
    if settings.get("test_interval"):
        for k in LOG_INTERVAL_KEYS:
            if k not in settings:
                settings[k] = settings["test_interval"]
                sources[k] = "test_interval"

    concession = {k[len("concession."):]: k for k in settings if k.startswith("concession.")}
    if not settings.get("budget.enabled"):
        for k in [k for k in settings if k.startswith(("budget.", "concession."))]:
            del settings[k], sources[k]
    elif concession:
        if "model" not in concession:
            raise ValueError(f"{sweep.name} {axes}: concession settings need concession.model")
        model_key = concession.pop("model")
        model = settings.pop(model_key)
        settings["budget.concession"] = model
        sources["budget.concession"] = sources.pop(model_key)
        for rest, key in concession.items():
            target = f"budget.concession_args.{model}.{rest}"
            settings[target] = settings.pop(key)
            sources[target] = sources.pop(key)

    for k, v in (overrides or {}).items():
        settings[k] = v
        sources[k] = "--set"
    settings["seed"] = seed
    sources["seed"] = "seeds"

    if settings.get("budget.enabled"):
        rule = settings.get("budget.rule", _default_budget_rule())
        launch = {"env_config": "gymma_budget",
                  "args": {"runner": f"budget_{_alg_runner(alg_name)}",
                           "common_reward": _rule_is_common(rule)}}
    else:
        launch = {"env_config": "gymma", "args": {}}
    return Run(sweep.name, dict(axes), seed, settings, sources, launch,
               settings_hash(settings, launch), dict(overrides or {}))


def _identity(tag):
    """What makes two tagged runs the same run of a sweep, settings aside:
    sweep, axis options, seed and any --set overrides."""
    return json.dumps([tag["sweep"], tag.get("axes"), tag.get("seed"), tag.get("set") or {}],
                      sort_keys=True)


# --- expanding a sweep ---------------------------------------------------

def _matches(axes_and_seed, condition):
    for axis, wanted in condition.items():
        wanted = wanted if isinstance(wanted, list) else [wanted]
        if axes_and_seed.get(axis) not in wanted and str(axes_and_seed.get(axis)) not in map(str, wanted):
            return False
    return True


def expand(sweep, where=None, seeds=None, overrides=None):
    """Every Run of the sweep, minus exclusions, filtered by where {axis: [options]}."""
    where = where or {}
    unknown = set(where) - set(sweep.axes) - {"seed"}
    if unknown:
        raise ValueError(f"unknown axes {sorted(unknown)} (axes: {list(sweep.axes)}, seed)")
    for axis, options in where.items():
        if axis != "seed":
            bad = set(options) - set(sweep.axes[axis])
            if bad:
                raise ValueError(f"axis '{axis}' has no options {sorted(bad)} "
                                 f"(available: {list(sweep.axes[axis])})")
    runs = []
    names = list(sweep.axes)
    for combo in itertools.product(*(sweep.axes[a] for a in names)):
        axes = dict(zip(names, combo))
        for seed in seeds if seeds is not None else sweep.seeds:
            point = {**axes, "seed": seed}
            if any(_matches(point, c) for c in sweep.exclude):
                continue
            if where and not _matches(point, where):
                continue
            runs.append(resolve(sweep, axes, seed, overrides))
    return runs


# --- what has run -------------------------------------------------------------

@dataclass
class Record:
    path: Path
    status: str
    tag: dict
    start: str
    heartbeat: str


def scan(sacred=SACRED):
    """Every sacred run launched by this engine (with a sweep tag)."""
    records = []
    for path in Path(sacred).glob("*/*/*/run.json"):
        try:
            info = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        comment = info.get("meta", {}).get("options", {}).get("--comment")
        try:
            tag = json.loads(comment) if comment else None
        except json.JSONDecodeError:
            continue
        if not isinstance(tag, dict) or "sweep" not in tag:
            continue
        records.append(Record(path.parent, info.get("status"), tag,
                              info.get("start_time") or "", info.get("heartbeat") or ""))
    return records


def _alive(record):
    if record.status != "RUNNING":
        return False
    try:
        beat = datetime.fromisoformat(record.heartbeat)
    except ValueError:
        return False
    return datetime.utcnow() - beat < STALE_HEARTBEAT


def states(runs, records=None):
    """[(run, state, record)] with state done / running / stale / failed / missing.
    record is the run's latest relevant sacred run, or None."""
    records = scan() if records is None else records
    by_hash, by_identity = {}, {}
    for r in sorted(records, key=lambda r: r.start):
        by_hash.setdefault((r.tag.get("hash"), r.tag.get("seed")), []).append(r)
        by_identity.setdefault(_identity(r.tag), []).append(r)

    out = []
    for run in runs:
        same = by_hash.get((run.hash, run.seed), [])
        done = [r for r in same if r.status == "COMPLETED"]
        if done:
            out.append((run, "done", done[-1]))
            continue
        alive = [r for r in same if _alive(r)]
        if alive:
            out.append((run, "running", alive[-1]))
            continue
        stale = [r for r in by_identity.get(run.identity, []) if r.status == "COMPLETED"]
        if stale:
            out.append((run, "stale", stale[-1]))
        elif same:
            out.append((run, "failed", same[-1]))
        else:
            out.append((run, "missing", None))
    return out


# --- notebook API ---------------------------------------------------------

def runs(sweep, seeds=None, overrides=None, **where):
    """One row per run of a sweep: its axis options, seed, state and sacred
    location (path is None until it has run). overrides is --set's dict.

        runs("lbf_budget", rule="increment")
    """
    import pandas as pd
    s = load_sweep(sweep)
    where = {k: v if isinstance(v, list) else [v] for k, v in where.items()}
    rows = []
    for run, state, record in states(expand(s, where, seeds, overrides)):
        rows.append({**run.axes, "seed": run.seed, "state": state,
                     "run": int(record.path.name) if record else None,
                     "path": record.path if record else None, "hash": run.hash})
    return pd.DataFrame(rows)


def load_runs(sweep, keys, seeds=None, overrides=None, states_=("done",), skip_missing=False,
              dropna=False, **where):
    """Training curves of a sweep's runs in load_results' long format, with a
    column per axis. Only done runs unless states_ says otherwise."""
    import pandas as pd
    import load_results
    table = runs(sweep, seeds, overrides, **where)
    frames = []
    for row in table[table.state.isin(states_)].to_dict("records"):
        path = row.pop("path")
        frame = load_results._metrics_frame(path, keys, dropna=dropna, skip_missing=skip_missing)
        frames.append(frame.assign(**{k: v for k, v in row.items() if k != "hash"}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --- launching --------------------------------------------------------------

def _log_path(run):
    safe = re.sub(r"[^A-Za-z0-9_.,=-]", "-", run.label)
    return LOGS / run.sweep / f"{safe}.log"


def run_one(run):
    """Train one run; returns the subprocess return code."""
    log_path = _log_path(run)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"PYTHONPATH": str(PROJECT), "OMP_NUM_THREADS": "1"}
    with open(log_path, "w") as log:
        proc = subprocess.run(run.command(), cwd=EPYMARL, env=env,
                              stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def launch(todo, max_workers=None):
    """Run every Run in todo in parallel; returns the failed ones."""
    max_workers = max_workers or max(1, (os.cpu_count() or 2) // 2)
    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_one, run): run for run in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            run, rc = futures[fut], fut.result()
            print(f"[{i}/{len(todo)}] {run.label}: {'ok' if rc == 0 else f'FAILED rc={rc}'}",
                  flush=True)
            if rc != 0:
                failures.append(run)
    return failures


# --- CLI ---------------------------------------------------------------------

def _parse_where(items):
    """["rule=increment,signed", "seed=0"] -> {"rule": ["increment", "signed"], "seed": [0]}"""
    where = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"filters look like axis=option[,option...], got '{item}'")
        axis, values = item.split("=", 1)
        values = values.split(",")
        where[axis] = [int(v) for v in values] if axis == "seed" else values
    return where


def _parse_set(items):
    """["t_max=300", "budget.initial_budget=[1.0, 6.0, 6.0]"] -> {key: YAML-parsed value}"""
    overrides = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set takes key=value, got '{item}'")
        key, value = item.split("=", 1)
        overrides[key] = yaml.safe_load(value)
    return overrides


def _expand_args(args):
    s = load_sweep(args.sweep)
    where = _parse_where(args.where)
    return s, where, _parse_set(args.set)


def _counts(rows):
    counts = {}
    for _, state, _ in rows:
        counts[state] = counts.get(state, 0) + 1
    order = ["done", "running", "stale", "failed", "missing"]
    return ", ".join(f"{counts[s]} {s}" for s in order if s in counts)


def _cmd_list(args):
    for name in sweep_names():
        s = load_sweep(name)
        sizes = " x ".join(f"{a}({len(o)})" for a, o in s.axes.items())
        print(f"{name:20s} {sizes} x seeds({len(s.seeds)})")


def _cmd_status(args):
    s, where, overrides = _expand_args(args)
    rows = states(expand(s, where, overrides=overrides))
    print(f"{s.name}: {len(rows)} runs: {_counts(rows)}")
    if args.list:
        for run, state, record in rows:
            where = f"  {record.path.relative_to(SACRED)}" if record else ""
            print(f"  {state:8s} {run.label}{where}")


def _cmd_show(args):
    s, where, overrides = _expand_args(args)
    where.setdefault("seed", [s.seeds[0]])
    matches = expand(s, where, overrides=overrides)
    if len(matches) != 1:
        free = [a for a in s.axes if a not in where]
        raise SystemExit(f"{len(matches)} runs match; narrow with axis=option for {free}")
    run = matches[0]
    print(f"{run.sweep}  {run.label}\nhash {run.hash}\n")
    width = max(len(k) for k in run.settings)
    for k in sorted(run.settings):
        print(f"  {k:{width}s} = {run.settings[k]!r:30s} [{run.sources[k]}]")
    print(f"\nlaunch: --env-config={run.launch['env_config']} {run.launch['args']}")
    print("\n" + " ".join(run.command()))


def _cmd_run(args):
    s, where, overrides = _expand_args(args)
    rows = states(expand(s, where, overrides=overrides))
    rerun = {"missing", "failed"} | ({"stale"} if args.rerun_stale else set())
    todo = [run for run, state, _ in rows if state in rerun]
    print(f"{s.name}: {len(rows)} runs: {_counts(rows)}; {len(todo)} to train", flush=True)
    stale = sum(state == "stale" for _, state, _ in rows)
    if stale and not args.rerun_stale:
        print(f"  {stale} stale (settings changed since they ran) left alone; "
              "--rerun-stale trains them again")
    if not todo:
        return 0
    if args.dry_run:
        for run in todo:
            print(f"  {run.label}")
        return 0
    if len(todo) > CONFIRM_ABOVE and not args.yes:
        print(f"refusing to launch {len(todo)} runs without --yes "
              "(narrow with axis=option filters, or check with --dry-run)")
        return 1
    failures = launch(todo, args.max_workers)
    if failures:
        print(f"{len(failures)} runs failed (logs in {LOGS / s.name})", file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python scripts/sweeps",
                                     description="YAML sweeps over epymarl (scripts/sweeps/*.yaml)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="sweeps and their axes")

    def sweep_parser(name, help):
        p = sub.add_parser(name, help=help)
        p.add_argument("sweep")
        p.add_argument("where", nargs="*", help="axis=option[,option...] filters, incl. seed=")
        p.add_argument("--set", nargs="+", metavar="KEY=VALUE",
                       help="final overrides (YAML values); they change the runs' hash, "
                            "so e.g. a smoke test never counts as the real run")
        return p

    p = sweep_parser("status", "done / running / stale / failed / missing counts")
    p.add_argument("--list", action="store_true", help="one line per run")
    sweep_parser("show", "one run's final settings and where each came from")
    p = sweep_parser("run", "train every missing or failed run")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rerun-stale", action="store_true")
    p.add_argument("--yes", action="store_true", help=f"allow more than {CONFIRM_ABOVE} runs")
    p.add_argument("--max-workers", type=int, default=None,
                   help="concurrent training subprocesses (default: half the cores)")
    args = parser.parse_args(argv)
    commands = {"list": _cmd_list, "status": _cmd_status, "show": _cmd_show, "run": _cmd_run}
    return commands[args.cmd](args) or 0
