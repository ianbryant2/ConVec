# PAC action scaling 
Training runs through [epymarl](https://github.com/uoe-agents/epymarl) so every
algorithm trains exactly as shipped, with additional algorithms (QTRAN, QPLEX,
weighted QMIX, optimistic exploration) ported faithfully from their original
repositories. See [epymarl/VENDORED.md](epymarl/VENDORED.md) for provenance
and the full list of local changes.

All baselines follow the Pareto-AC paper's setup: non-parameter-shared (`_ns`)
variants throughout.

## Layout

| Path | What it is |
|---|---|
| [matrix_envs/games.py](matrix_envs/games.py) | Normal-form game definitions, one per failure mode |
| [matrix_envs/](matrix_envs/) | Registers each game as a one-step gymnasium env for epymarl's `gymma` wrapper |
| [games_sweep.ipynb](games_sweep.ipynb) | The main sweep (algorithms × games × seeds) plots and results|
| [set_eval/set_evaluators.py](set_eval/set_evaluators.py), [test_set_evals.ipynb](test_set_evals.ipynb) | Tabular/factored set evaluators (expectile/quantile fits) and their tests |
| [set_eval/dataset_builder.py](set_eval/dataset_builder.py) | Synthetic value-table distributions (pairwise/k-wise structure) |
| [epymarl/](epymarl/) | Vendored epymarl with the ported algorithms ([VENDORED.md](epymarl/VENDORED.md)) |
| [scripts/sweeps/](scripts/sweeps/) | YAML sweeps: each `<sweep>.yaml` crosses axes (alg × env × any others × seeds) built from [presets/](scripts/sweeps/presets/); `python scripts/sweeps list / status / show / run <sweep> [axis=option ...]`. Format and merge order in [engine.py](scripts/sweeps/engine.py). The old `run_sweep.py` is at git tag `sweep-legacy` |
| [scripts/load_results.py](scripts/load_results.py) | Sacred runs → one long DataFrame, a row per (run, evaluation point) |
| [scripts/run_catalog.yaml](scripts/run_catalog.yaml) | Every completed sacred run, grouped by the settings it was launched with: a label, run numbers, seeds and the exact overrides per group. Look up a run here; rebuild and query with [scripts/run_catalog.py](scripts/run_catalog.py) |
| [scripts/plotting.py](scripts/plotting.py) | That DataFrame → one panel per call; notebooks only choose the layout |


## Setup

```bash
scripts/install.sh    # creates .venv
```
