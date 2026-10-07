# PAC action scaling: concession budgets

Multi-agent RL with a per-agent **concession budget**: each step, agent $i$ is
charged its concession $\delta_i = V^*_i(s) - Q^*_i(s, a)$, how much the joint
action costs it against the best it could get, and the budget rule rewards the
team for keeping every agent within its budget $c_i$. $Q^*_i$ comes from a
learned critic trained alongside the agents.

Training runs through a vendored [epymarl](https://github.com/uoe-agents/epymarl);
see [epymarl/VENDORED.md](epymarl/VENDORED.md) for provenance and local changes.
All algorithms are the non-parameter-shared (`_ns`) variants.

## Results

| Sweep | Env | Algorithms | Seeds |
|---|---|---|---|
| [matrix_table1](scripts/sweeps/matrix_table1.yaml) | Budget Table 1: one-step 3-agent game, $c = (6, 6, 5)$, only $(A, A, A)$ within budget | PAC, MAPPO, VDN, QMIX | 5 |
| [lbf_td_converge](scripts/sweeps/lbf_td_converge.yaml) | Level-Based Foraging 5x5, 3 agents, 2 apples, $c = (0.55, 0.8, 0.8)$ | PAC (7M steps) | 5 |
| [selfish_spread_race](scripts/sweeps/selfish_spread_race.yaml) | SelfishSpread race: 3 agents race for the first of a row of landmarks, $c = (3, 20, 20)$ | PAC (3M steps) | 5 |

All use the same concession critic and budget rule:

- **Critic**: `optimq_additive_mlp`, with state and per-agent action
  embeddings added, then one hidden layer. It is trained only on its own
  exploration episodes, epsilon-greedy on its own $Q^*$ heads, and frozen once
  exploration has annealed to zero, so the agents learn against fixed charges.
- **Charge** (`delta: td`): $V(s_t) - r_t - \gamma V(s_{t+1})$, whose
  discounted episode total telescopes to $V(s_0) - G$.
- **Rule**: `joint_overspend_potential`.

The settings are in [scripts/sweeps/presets/](scripts/sweeps/presets/): the
critic in `concession.yaml`, per-algorithm hyperparameters in `algs.yaml`,
envs in `envs.yaml`.

## Setup

Python 3.10.

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt    # includes ./lb-foraging, editable
```

## Reproducing

```bash
.venv/bin/python scripts/sweeps run matrix_table1
.venv/bin/python scripts/sweeps run lbf_td_converge
.venv/bin/python scripts/sweeps run selfish_spread_race

.venv/bin/python scripts/sweeps status <sweep> --list    # which runs are done, and their sacred dirs
.venv/bin/python scripts/sweeps show <sweep> alg=<alg>   # a run's resolved settings and command
```

Runs go to `epymarl/results/sacred/<alg name>/<env key>/<run>/` (`config.json`,
`metrics.json`, and for the learned critic `concession_critic.pt`). A run counts
as done when a completed run has the same settings hash and seed.
[scripts/load_results.py](scripts/load_results.py) loads them into one long
DataFrame, and `engine.load_runs(<sweep>, keys)` loads a sweep's runs with
their axis options as columns.

## Layout

| Path | What it is |
|---|---|
| [epymarl/src/runners/budget_runner.py](epymarl/src/runners/budget_runner.py) | The budget runners: epymarl's episode and parallel runners with the concession charged and the budget applied each step |
| [epymarl/src/envs/budget/](epymarl/src/envs/budget/) | Budget bookkeeping and the reward rules |
| [epymarl/src/components/](epymarl/src/components/) | The concession critic (`optimq.py`, `optimq_additive.py`), the charge (`budget_concession.py`) and its training data (`concession_training.py`) |
| [epymarl/src/config/envs/gymma_budget.yaml](epymarl/src/config/envs/gymma_budget.yaml) | Every budget and critic option, with defaults |
| [matrix_envs/](matrix_envs/) | One-step matrix games as gymnasium envs, including Budget Table 1 (`budget_table1` in [games.py](matrix_envs/games.py)) |
| [lbf_envs/](lbf_envs/), [lb-foraging/](lb-foraging/) | The LBF budget env, on a vendored lb-foraging |
| [mpe_envs/](mpe_envs/) | The SelfishSpread race: PettingZoo's simple_spread with individual rewards and a race layout |
| [scripts/sweeps/](scripts/sweeps/) | The sweeps and their presets; format and merge order in [engine.py](scripts/sweeps/engine.py) |
