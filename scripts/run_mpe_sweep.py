"""MPE simple_spread sweep: 4 non-shared algs x N in {3,6,8} x 3 seeds, 1M steps.

Uses the *_mpe config variants (qmix_ns_mpe, qplex_ns_mpe, opt_qmix_ns_mpe,
pac_dcg_ns_mpe): each starts from its original alg but swaps in
hyperparameters transferred from EPyMARL's own MPE hyperparameter search
(arXiv:2006.07869 Table 29, for the QMIX-family algs) or fixes settings that
were leftover matrix-game/SMAC conventions never suited to a 25-step episodic
env (opt_qmix's buffer=32/target=1, pac_dcg's equilibrium-selection entropy
anneal). See each yaml's header comment for the specific rationale. This is
a fixed-hyperparameter re-run of the original baseline sweep, not a search.
Note qmix/qplex/opt_qmix now use GRU (roughly 2x per-step cost vs the
original FC baseline sweep), so expect wall-clock above that sweep's ~6h.

Run from the repo root (resume-safe; rerun after interruption to continue):

    nohup .venv/bin/python scripts/run_mpe_sweep.py > mpe_sweep.log 2>&1 &
"""

import importlib.util
import sys
from pathlib import Path

# Every training run is a subprocess of this same interpreter; bail out up
# front if it isn't the project venv (e.g. launched without activating .venv).
if importlib.util.find_spec("torch") is None:
    sys.exit(f"{sys.executable} can't import torch — run this with the project "
             "venv, e.g.: .venv/bin/python scripts/run_mpe_sweep.py")

_spec = importlib.util.spec_from_file_location(
    "run_sweep", Path.cwd() / "scripts" / "run_sweep.py")
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

ALGS = ["pac_dcg_ns_mpe", "qmix_ns_mpe", "qplex_ns_mpe", "opt_qmix_ns_mpe"]
ENVS = [f"mpe_envs:SimpleSpread-{n}ag-v0" for n in (3, 6, 8)]

failures = sweep.run_sweep(algs=ALGS, envs=ENVS, t_max=1_000_000,
                           test_interval=25_000, test_nepisode=100,
                           time_limit=25, max_workers=12)
print("FAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
