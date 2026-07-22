"""Sweep epymarl training over the matrix-game (and MPE) envs.

Each run is one epymarl/src/main.py subprocess; runs with a COMPLETED sacred
record are skipped, so the sweep is resume-safe. Run via scripts/run.sh or
import from games_sweep.ipynb. cwd must be the repo root either way.

Env entries are bare matrix_envs ids by default; an entry containing ":" is
used as a full gymma key, e.g. "mpe_envs:SimpleSpread-4ag-v0" (see MPE_ENVS).
Multi-step envs need time_limit > 1, e.g.
run_sweep(envs=MPE_ENVS, time_limit=25, t_max=2_050_000).
"""

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

# default grid for standalone runs; games_sweep.ipynb passes its own.
# all non-shared (_ns) variants, matching the Pareto-AC paper's baseline setup
ALGS = ["iql_ns", "vdn_ns", "qmix_ns", "qtran_ns", "pac_ns", "pac_dcg_ns",
        "opt_qmix_ns", "opt_vdn_ns", "cw_qmix_ns", "ow_qmix_ns", "qplex_ns"]

ENVS = (
    [f"risky-coordination-{n}p-v0" for n in (2, 3, 4)]
    + [f"climbing-{n}p-v0" for n in (2, 3, 4)]
    + [f"penalty-{k}-2p-v0" for k in (0, 25, 50, 75, 100)]
    + ["penalty-100-3p-v0"]
    + [f"anti-coordination-{n}p-v0" for n in (2, 3, 4)]
    + [f"needle-{n}p-v0" for n in (2, 3, 4)]
    + [f"kwise-o{order}-t4-4p-s{s}-v0" for order in (2, 3) for s in (0, 1, 2)]
)

# MPE simple_spread at increasing team sizes (see mpe_envs/__init__.py);
# episodes are 25 steps, so pass time_limit=25 and a larger t_max.
MPE_ENVS = [f"mpe_envs:SimpleSpread-{n}ag-v0" for n in (3, 4, 5, 6, 8)]

SEEDS = [0, 1, 2]

TIME_LIMIT = 1           # steps per episode (matrix games are one-step)
T_MAX = 50_000          # one-step episodes per run
TEST_INTERVAL = 1_000    # -> ~100 greedy evaluation points per curve
TEST_NEPISODE = 20       # greedy test episodes per evaluation point
EXTRA_OVERRIDES = []     # extra sacred overrides, e.g. ["common_reward=False"]


def sacred_name(alg):
    """Sacred names its results dir after the yaml's experiment name, which for the
    PAC configs differs from the --config name (pac_ns -> pac_sarsa_ns)."""
    text = (EPYMARL / "src" / "config" / "algs" / f"{alg}.yaml").read_text()
    return re.search(r'^name:\s*"?([^"\s]+)"?', text, re.M).group(1)


def full_key(env_id):
    """Gymma env key for an ENVS entry: bare ids are matrix_envs games."""
    return env_id if ":" in env_id else f"matrix_envs:{env_id}"


def overrides(env_id, seed, t_max=T_MAX, test_interval=TEST_INTERVAL,
              test_nepisode=TEST_NEPISODE, extra_overrides=EXTRA_OVERRIDES,
              time_limit=TIME_LIMIT):
    return [
        f"env_args.time_limit={time_limit}",
        f"env_args.key={full_key(env_id)}",
        "reward_scalarisation=mean",  # logged return == game-table value
        f"t_max={t_max}",
        f"test_interval={test_interval}",
        f"test_nepisode={test_nepisode}",
        f"log_interval={test_interval}",
        f"runner_log_interval={test_interval}",
        f"learner_log_interval={test_interval}",
        "use_cuda=False",
        f"seed={seed}",
    ] + list(extra_overrides)


def iter_run_dirs(algs=ALGS):
    """Yield (alg config name, env_id, run_dir) for every numbered sacred run dir."""
    sacred_to_alg = {sacred_name(a): a for a in algs}
    for alg_dir in SACRED.glob("*/"):
        alg = sacred_to_alg.get(alg_dir.name, alg_dir.name)
        for key_dir in alg_dir.glob("*:*/"):
            env_id = (key_dir.name.split(":", 1)[1]
                      if key_dir.name.startswith("matrix_envs:") else key_dir.name)
            for run_dir in key_dir.iterdir():
                if run_dir.name.isdigit():
                    yield alg, env_id, run_dir


def completed_runs(algs=ALGS, t_max=T_MAX, time_limit=TIME_LIMIT):
    """(alg, env_id, seed) -> run_dir for every COMPLETED run matching the sweep settings."""
    done = {}
    for alg, env_id, run_dir in iter_run_dirs(algs):
        try:
            config = json.loads((run_dir / "config.json").read_text())
            status = json.loads((run_dir / "run.json").read_text())["status"]
        except (OSError, KeyError, json.JSONDecodeError):
            continue
        if (status == "COMPLETED" and config.get("t_max") == t_max
                and config.get("env_args", {}).get("time_limit") == time_limit):
            # Later run numbers win if a combination was trained twice.
            done[(alg, env_id, config["seed"])] = run_dir
    return done


def run_one(alg, env_id, seed, t_max=T_MAX, test_interval=TEST_INTERVAL,
            test_nepisode=TEST_NEPISODE, extra_overrides=EXTRA_OVERRIDES,
            time_limit=TIME_LIMIT):
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{alg}__{env_id.replace(':', '-')}__s{seed}.log"
    cmd = [sys.executable, "src/main.py", f"--config={alg}", "--env-config=gymma",
           "with"] + overrides(env_id, seed, t_max, test_interval, test_nepisode,
                                extra_overrides, time_limit)
    env = os.environ | {"PYTHONPATH": str(PROJECT), "OMP_NUM_THREADS": "1"}
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=EPYMARL, env=env,
                              stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def run_sweep(algs=ALGS, envs=ENVS, seeds=SEEDS, t_max=T_MAX, test_interval=TEST_INTERVAL,
              test_nepisode=TEST_NEPISODE, extra_overrides=EXTRA_OVERRIDES,
              max_workers=None, time_limit=TIME_LIMIT):
    """Train every missing (alg, env, seed) in the grid; skip what's already done.

    Returns the list of (alg, env_id, seed) runs that failed (empty if all ok)."""
    max_workers = max_workers or max(1, (os.cpu_count() or 2) // 2)
    done = completed_runs(algs, t_max, time_limit)
    todo = [(a, e, s) for a in algs for e in envs for s in seeds
            if (a, e, s) not in done]
    print(f"{len(done)} runs already complete, {len(todo)} to train "
          f"({max_workers} workers)", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_one, a, e, s, t_max, test_interval,
                                test_nepisode, extra_overrides, time_limit): (a, e, s)
                   for a, e, s in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            run, rc = futures[fut], fut.result()
            print(f"[{i}/{len(todo)}] {run[0]} / {run[1]} / seed {run[2]}: "
                  f"{'ok' if rc == 0 else f'FAILED rc={rc}'}", flush=True)
            if rc != 0:
                failures.append(run)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-workers", type=int, default=None,
                        help="concurrent training subprocesses "
                             "(default: half the cores; each run is single-threaded)")
    args = parser.parse_args()

    failures = run_sweep(max_workers=args.max_workers)
    if failures:
        print(f"{len(failures)} runs failed (logs in {LOGS}): {failures}",
              file=sys.stderr)
        return 1
    print("sweep complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
