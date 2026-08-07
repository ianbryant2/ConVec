import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT = Path.cwd()
EPYMARL = PROJECT / "epymarl"
SACRED = EPYMARL / "results" / "sacred"
LOGS = EPYMARL / "results" / "sweep_logs"

ALGS = ["pac_ns"]  # config file/sacred name is pac_max_ns (pac_sarsa_max_ns)

# Level-Based Foraging, registered by lb-foraging/lbforaging/__init__.py.
# Bare ids (no ":") are matrix_envs games, e.g. "climbing-3p-v0".
ENVS = [
    "lbforaging:Foraging-5x5-2p-1f-pen-v3",
]

SEEDS = [0, 1]

# Applied to every run. Matrix-game defaults: one-step episodes, small budget.
BASE_OVERRIDES = [
    {
        "env_args.time_limit": 1,     # steps per episode (matrix games are one-step)
        "t_max": 50_000,              # environment steps per run
        "test_interval": 1_000,       # -> ~50 greedy evaluation points per curve
        "test_nepisode": 20,          # greedy test episodes per evaluation point
    },
    {
        "reward_scalarisation": "sum",  # logged return == game-table value
        "use_cuda": False,
    },
]

# Per-algorithm settings, e.g. {"pac_max_ns": [{"q_nstep": 5, "lr": 0.0005}]}.
ALG_OVERRIDES = {
    "pac_max_ns": [],
}

# Per-environment settings. LBF/MPE episodes are 25 steps and need a much
# larger budget than the one-step matrix games in BASE_OVERRIDES.
ENV_OVERRIDES = {
    "lbforaging:Foraging-5x5-2p-1f-pen-v3": [
        {"env_args.time_limit": 25, "t_max": 14_000_000, "test_interval": 140_000, 'test_greedy' : False},
    ]
}

# Mirrored from test_interval unless a dict above sets them explicitly.
LOG_INTERVAL_KEYS = ["log_interval", "runner_log_interval", "learner_log_interval"]


def sacred_name(alg):
    """Sacred names its results dir after the yaml's experiment name, which for the
    PAC configs differs from the --config name (pac_ns -> pac_sarsa_ns)."""
    text = (EPYMARL / "src" / "config" / "algs" / f"{alg}.yaml").read_text()
    return re.search(r'^name:\s*"?([^"\s]+)"?', text, re.M).group(1)


def overrides(alg, env, seed):
    """{sacred arg: value} for one run, merging base, alg and env overrides."""
    merged = {}
    for group in (BASE_OVERRIDES + ALG_OVERRIDES.get(alg, [])
                  + ENV_OVERRIDES.get(env, [])):
        merged.update(group)
    merged["env_args.key"] = env
    merged["seed"] = seed
    if merged.get("test_interval"):
        for key in LOG_INTERVAL_KEYS:
            merged.setdefault(key, merged["test_interval"])
    return merged


def iter_run_dirs(algs=ALGS):
    """Yield (alg config name, env id, run_dir) for every numbered sacred run dir."""
    sacred_to_alg = {sacred_name(a): a for a in algs}
    for alg_dir in SACRED.glob("*/"):
        alg = sacred_to_alg.get(alg_dir.name, alg_dir.name)
        for key_dir in alg_dir.glob("*:*/"):
            env = (key_dir.name.split(":", 1)[1]
                   if key_dir.name.startswith("matrix_envs:") else key_dir.name)
            for run_dir in key_dir.iterdir():
                if run_dir.name.isdigit():
                    yield alg, env, run_dir


def _completed(algs):
    """Yield (alg, env id, seed, config, run_dir) for every COMPLETED sacred run."""
    for alg, env, run_dir in iter_run_dirs(algs):
        try:
            config = json.loads((run_dir / "config.json").read_text())
            status = json.loads((run_dir / "run.json").read_text())["status"]
        except (OSError, KeyError, json.JSONDecodeError):
            continue
        if status == "COMPLETED":
            yield alg, env, config["seed"], config, run_dir


def _matches(config, merged):
    """True if a sacred config was trained with all of merged's settings."""
    for key, value in merged.items():
        node = config
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        if node != value and str(node) != str(value):
            return False
    return True


def completed_runs(algs=ALGS, envs=ENVS, seeds=SEEDS):
    """(alg, env, seed) -> run_dir for runs already trained with these overrides.

    Later run numbers win if a combination was trained twice."""
    wanted = {(a, e, s): overrides(a, e, s)
              for a in algs for e in envs for s in seeds}
    done = {}
    for alg, env, seed, config, run_dir in _completed(algs):
        run = (alg, env, seed)
        if run in wanted and _matches(config, wanted[run]):
            done[run] = run_dir
    return done


def run_one(alg, env, seed):
    """Train one (alg, env, seed); returns the subprocess return code."""
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{alg}__{env.replace(':', '-')}__s{seed}.log"
    args = [f"{k}={v}" for k, v in overrides(alg, env, seed).items()]
    cmd = [sys.executable, "src/main.py", f"--config={alg}", "--env-config=gymma",
           "with"] + args
    proc_env = os.environ | {"PYTHONPATH": str(PROJECT), "OMP_NUM_THREADS": "1"}
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=EPYMARL, env=proc_env,
                              stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def run_sweep(algs=ALGS, envs=ENVS, seeds=SEEDS, max_workers=None):
    """Train every missing (alg, env, seed) in the grid; skip what's already done.

    Returns the list of (alg, env, seed) runs that failed (empty if all ok)."""
    max_workers = max_workers or max(1, (os.cpu_count() or 2) // 2)
    done = completed_runs(algs, envs, seeds)
    todo = [(a, e, s) for a in algs for e in envs for s in seeds
            if (a, e, s) not in done]
    print(f"{len(done)} runs already complete, {len(todo)} to train "
          f"({max_workers} workers)", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_one, a, e, s): (a, e, s) for a, e, s in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            run, rc = futures[fut], fut.result()
            print(f"[{i}/{len(todo)}] {run[0]} / {run[1]} / seed {run[2]}: "
                  f"{'ok' if rc == 0 else f'FAILED rc={rc}'}", flush=True)
            if rc != 0:
                failures.append(run)
    return failures


def main():
    print(__doc__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=None,
                        help="concurrent training subprocesses "
                             "(default: half the cores; each run is single-threaded)")
    parser.add_argument("--algs", type=str, default=None,
                        help=f"comma-separated algorithms (default: {','.join(ALGS)})")
    parser.add_argument("--envs", type=str, default=None,
                        help=f"comma-separated environments (default: {','.join(ENVS)})")
    parser.add_argument("--seeds", type=str, default=None,
                        help="comma-separated seeds to train (default: "
                             f"{','.join(map(str, SEEDS))}). Runs are matched on "
                             "(alg, env, seed) plus their sacred config, so a "
                             "disjoint set (e.g. 3,4,5) adds new runs alongside the "
                             "existing ones instead of skipping them as already-done "
                             "-- handy for comparing before/after a code change like "
                             "the greedy-eval fix to SoftPoliciesSelector.")
    args = parser.parse_args()

    algs = args.algs.split(",") if args.algs else ALGS
    envs = args.envs.split(",") if args.envs else ENVS
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS

    failures = run_sweep(algs=algs, envs=envs, seeds=seeds,
                         max_workers=args.max_workers)
    if failures:
        print(f"{len(failures)} runs failed (logs in {LOGS}): {failures}",
              file=sys.stderr)
        return 1
    print("sweep complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
