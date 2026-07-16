# Vendored from uoe-agents/epymarl

- Upstream: https://github.com/uoe-agents/epymarl
- Base commit: `cbc38c09588064eab978501d0f12c2cf58fa7fc2` (2024-09-24, "Update README.md")
- Vendored into this repo on 2026-07-15; the upstream `.git` history was dropped.
  Changes after that date are tracked in this repo's history.

## Local changes relative to upstream at vendoring time

Modified:
- `src/components/action_selectors.py`
- `src/controllers/__init__.py`
- `src/envs/__init__.py` (smaclite import fix)
- `src/learners/__init__.py`
- `src/learners/qtran_learner.py`
- `src/modules/agents/__init__.py`

Added — algorithm ports, kept faithful to their original implementations:
- QTRAN configs: `src/config/algs/qtran.yaml`, `qtran_ns.yaml`
- QPLEX (from [wjh720/QPLEX](https://github.com/wjh720/QPLEX) @ `6c0a7ba`):
  `src/config/algs/qplex_ns.yaml`, `src/learners/dmaq_qatten_learner.py`,
  `src/modules/mixers/dmaq_general.py`, `src/modules/mixers/dmaq_si_weight.py`
- Weighted QMIX (CW/OW) and optimistic exploration (both ported from
  [qxqxtxdy/OptimisticExploration](https://github.com/qxqxtxdy/OptimisticExploration) @ `a7a8bf0`):
  `src/config/algs/cw_qmix_ns.yaml`, `ow_qmix_ns.yaml`, `opt_qmix_ns.yaml`, `opt_vdn_ns.yaml`,
  `src/learners/max_q_learner.py`, `src/learners/opt_q_learner.py`,
  `src/controllers/central_basic_controller.py`, `src/controllers/opt_controller.py`,
  `src/modules/agents/central_rnn_agent.py`, `src/modules/mixers/qmix_central_no_hyper.py`,
  `src/utils/normalize.py`
