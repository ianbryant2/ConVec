"""Sweep epymarl training over a grid of (algorithm, env, variant, seed).

Each run is one epymarl/src/main.py subprocess, built from the override dicts
below (BASE_OVERRIDES, ALG_OVERRIDES, ENV_OVERRIDES, then VARIANTS). Runs whose
COMPLETED sacred config already has those settings are skipped, so the sweep is
resume-safe. Run from the repo root, either as

    python scripts/run_sweep.py [--variants stable,slow --seeds 0,1 ...]

or by importing run_sweep() from a notebook. Env ids without ":" are
matrix_envs games; others are full gymma keys, e.g. "mpe_envs:SelfishSpread-3ag-v0".
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

# The algorithms from the budget-table1 sweep: PAC and OptVDN were the only ones
# to reach the optimum of both climbing-3p and risky-coordination-3p in the
# archived matrix sweep (games_sweep_archive.ipynb); QMIX and VDN are controls.
ALGS = ["pac_ns", "opt_vdn_ns", "qmix_ns", "vdn_ns"]
# OptimQ exploration sweep: one algorithm per runner (PAC: parallel, VDN:
# episode); VDN's OptimQ diverged fastest in the SelfishSpread runs.
# ALGS = ["pac_ns"]

# SelfishSpread with a wider collision distance: two agents within d count as
# collided in the reward (default 0.3, the sum of their sizes). Each d has its
# own env id, mpe_envs:SelfishSpread-3ag-col{d}-v0, and gets the same settings
# as the plain env (ENV_OVERRIDES below). d must be listed in
# mpe_envs.COLLISION_DISTANCES, where the ids are registered.
COLLISION_DISTANCES = [0.4, 0.6]

# Bare ids (no ":") are matrix_envs games, e.g. "climbing-3p-v0".
ENVS = [
    # "mpe_envs:SelfishSpread-3ag-v0",
    # *(f"mpe_envs:SelfishSpread-3ag-col{d:g}-v0" for d in COLLISION_DISTANCES),
    # "budget-table1-3p-v0",
    # 3 agents, 2 apples, all level 1 (lbf_envs/__init__.py); run with LBF_VARIANTS.
    "lbf_envs:Foraging-5x5-3p-2f-budget-v0",
]

SEEDS = [0, 1, 2]

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
# These apply to every env in ENVS.
#
# budget-table1 (one-step matrix game): the plain yaml defaults, as used by the
# archived matrix sweep and the completed budget-table1-3p-v0 runs.
_MATRIX_ALG_OVERRIDES = {
    "pac_ns": [{"initial_entropy_coef": 20.0, "final_entropy_coef": 0.01,
                "lr": 0.0003, "hidden_dim": 64, "buffer_size": 10,
                "target_update_interval_or_tau": 0.01, "use_rnn": False}],
    "opt_vdn_ns": [{"epsilon_anneal_time": 50_000, "normal_anneal_time": 50_000,
                    "lr": 0.0005, "hidden_dim": 64, "target_update_interval": 1,
                    "buffer_size": 32, "use_rnn": False}],
    "qmix_ns": [{"epsilon_anneal_time": 50_000, "lr": 0.0005, "hidden_dim": 64,
                 "target_update_interval_or_tau": 200, "buffer_size": 5000,
                 "use_rnn": False}],
    "vdn_ns": [{"epsilon_anneal_time": 50_000, "lr": 0.0005, "hidden_dim": 64,
                "target_update_interval_or_tau": 200, "buffer_size": 5000,
                "use_rnn": False}],
}

# SelfishSpread (25-step MPE episodes): each is the change from the plain yaml
# to its MPE-tuned sibling (pac_dcg_ns_mpe, qmix_ns_mpe, opt_qmix_ns_mpe; the
# plain VDN configs differ from their QMIX ones only in the mixer).
#
# OptimQ takes replay_ratio gradient steps per critic per new transition,
# whatever the runner. 0.08 is what the earlier per-rollout settings came to:
# 20 steps per 250-transition parallel rollout (PAC) and 2 per 25-transition
# episode rollout (the rest).
_OPTIMQ_MPE = {"budget.concession_args.optimq.training.replay_ratio": 0.08}
_OPTIMQ_DATA = "budget.concession_args.optimq.data."
_QMIX_MPE = {"epsilon_anneal_time": 200_000, "lr": 0.0003, "hidden_dim": 128,
             "target_update_interval_or_tau": 0.01, "use_rnn": True}
_MPE_ALG_OVERRIDES = {
    # Entropy floor 0.05 instead of 0.01: the LBF PAC runs collapsed as their
    # entropy reached its 0.02 floor.
    "pac_ns": [{"initial_entropy_coef": 0.8, "final_entropy_coef": 0.05,
                "lr": 0.0005, "hidden_dim": 128}, _OPTIMQ_MPE],
    "opt_vdn_ns": [{"epsilon_anneal_time": 200_000, "normal_anneal_time": 200_000,
                    "lr": 0.0003, "hidden_dim": 128, "target_update_interval": 200,
                    "buffer_size": 5000, "use_rnn": True}, _OPTIMQ_MPE],
    "qmix_ns": [_QMIX_MPE, _OPTIMQ_MPE],
    "vdn_ns": [_QMIX_MPE, _OPTIMQ_MPE],
}

# Level-Based Foraging: the published LBF settings. PAC is the Pareto-AC
# paper's "LBF 3 Ag." column (Christianos et al. 2023, Table 3, arXiv
# 2209.14344) with its n-step of 5 (Section 4.5); QMIX and VDN are EPyMARL's
# LBF settings without parameter sharing (Papoudakis et al. 2021, Tables 27
# and 29, arXiv 2006.07869). Pareto-AC tuned no baselines it reports.
# OptVDN has no published LBF settings: it takes VDN's (its plain yaml differs
# from vdn_ns only in the optimistic parts), with the optimistic weight's
# anneal matched to epsilon's as in the other override sets. It keeps its
# unstandardised rewards, since opt_q_learner ignores standardise_rewards.
#
# OptimQ on LBF (components/concession_training.py): a 100k-step random-play
# warmup, then the data set by the variant (_LBF_OWN / _LBF_EXPLORE_ONLY),
# counted per agent episode, so PAC's parallel runner (10 episodes per learner
# update) gives its OptimQ 10x the data and steps of the episode runner's.
# _STABLE's fixes, with LBF's reward range (an apple pays at most 1; the
# SelfishSpread [-8, 0] would clamp every positive target to 0).
_LBF_OPTIMQ = {"budget.concession": "optimq",
               f"{_OPTIMQ_DATA}explore_policy": "random",
               f"{_OPTIMQ_DATA}warmup_steps": 100_000,
               # The old per-iteration setting: 8 steps per 4 new 25-step episodes.
               "budget.concession_args.optimq.training.replay_ratio": 0.08,
               "budget.concession_args.optimq.target_tau": 0.005,
               "budget.concession_args.optimq.loss": "huber",
               "budget.concession_args.optimq.layer_norm": True,
               "budget.concession_args.optimq.reward_range": [0.0, 1.0]}
_VDN_LBF = {"epsilon_anneal_time": 50_000, "evaluation_epsilon": 0.05, "lr": 0.0001,
            "hidden_dim": 64, "buffer_size": 5000, "standardise_rewards": True,
            "use_rnn": True}
_LBF_ALG_OVERRIDES = {
    "pac_ns": [{"initial_entropy_coef": 0.8, "final_entropy_coef": 0.02,
                "entropy_end_ratio": 1.0, "lr": 0.0003, "hidden_dim": 128,
                "q_nstep": 5, "target_update_interval_or_tau": 0.01,
                # Evaluation samples the policy, as both papers do for
                # stochastic-policy algorithms (pac_ns_lbf.yaml does too).
                "test_greedy": False,
                "use_rnn": False}, _LBF_OPTIMQ],
    "qmix_ns": [_VDN_LBF, {"target_update_interval_or_tau": 0.01}, _LBF_OPTIMQ],
    "vdn_ns": [_VDN_LBF, {"target_update_interval_or_tau": 200}, _LBF_OPTIMQ],
    "opt_vdn_ns": [{"epsilon_anneal_time": 50_000, "normal_anneal_time": 50_000,
                    "evaluation_epsilon": 0.05, "lr": 0.0001, "hidden_dim": 64,
                    "buffer_size": 5000, "target_update_interval": 200,
                    "use_rnn": True}, _LBF_OPTIMQ],
}
LBF_PREFIXES = ("lbforaging:", "lbf_envs:")

# Both papers give every algorithm the same number of gradient updates, not
# the same environment steps: PAC's parallel runner makes one update per 10
# episodes, the episode runner one per episode, so PAC gets 10x the steps
# (e.g. 20M vs 2M in EPyMARL). On LBF envs these env-step keys are multiplied
# by the alg's factor after all overrides are merged, so t_max set in
# ENV_OVERRIDES is the value-based budget and evaluation points line up.
# OptimQ's data is counted per agent episode, so it follows the scaling; the
# warmup is a fixed buffer fill. Other envs keep equal steps, so their
# completed runs still match.
T_MAX_SCALE = {"pac_ns": 10}
_SCALED_STEP_KEYS = ["t_max", "test_interval"]

# Used for every other env with ":" in its id; bare matrix_envs ids always get
# _MATRIX_ALG_OVERRIDES and LBF ids _LBF_ALG_OVERRIDES (see overrides()), so
# one sweep can hold all three.
ALG_OVERRIDES = _MPE_ALG_OVERRIDES

# Set explicitly, not left to default.yaml: completed runs are matched on the
# keys set here, so runs made with the old "joint_overspend" rule (which
# charged the overspend level every step) are not mistaken for these.
_BUDGET_RULE = "joint_budget_signed_increment"

# SelfishSpread, 3 agents: agent 0 has priority (budget 1, about a typical
# conflict), the others give way (6 each, above the ~5.7 90th percentile of
# the largest per-episode concession). No payoff table, so delta comes from
# the learned OptimQ critics. 25-step episodes and 1M steps as in the
# archived MPE sweep.
# 300k steps: OptimQ diverged within 25k (VDN) and 150k (PAC) steps in the
# 1M-step runs, so this is enough to see whether a variant stays stable.
# OptimQ's replay_ratio is set in _MPE_ALG_OVERRIDES.
_SELFISH_SPREAD_3AG = [
    {"env_args.time_limit": 25, "t_max": 300_000, "test_interval": 25_000},
    {"budget.enabled": True, "budget.initial_budget": [1.0, 50.0, 50.0],
     "budget.concession": "optimq", "budget.rule": _BUDGET_RULE},
]

# Per-environment settings. The concession budget is switched on here too.
ENV_OVERRIDES = {
    # Table 1 with c = (6, 6, 5): only (A, A, A) stays within budget. Exact delta
    # from the payoff table (budget.concession defaults to "utopian_table").
    "budget-table1-3p-v0": [
        {"budget.enabled": True, "budget.initial_budget": [6.0, 6.0, 5.0], "budget.concession" : "optimq",
         "budget.rule": _BUDGET_RULE},
    ],
    "mpe_envs:SelfishSpread-3ag-v0": _SELFISH_SPREAD_3AG,
    # Wider collision distances: same settings; only the env id differs.
    **{f"mpe_envs:SelfishSpread-3ag-col{d:g}-v0": _SELFISH_SPREAD_3AG
       for d in COLLISION_DISTANCES},
    # LBF budget test (lbf_envs/__init__.py): agent 0 should eat one apple alone
    # and agents 1 and 2 share the other, conceding about [0.47, 0.75, 0.75]
    # (apples pay 1/2, split among the agents loading them). The budget allows
    # that and rules out the alternatives; the closest, all three sharing the
    # second apple after agent 0 eats the first, sits right at 0.8 for agents
    # 1 and 2. t_max is the value-based budget (PAC's is 10x, T_MAX_SCALE): PAC
    # trains 14M steps as in the Pareto-AC paper's LBF runs (its plots run to
    # 1.4e7; EPyMARL used 20M / 2M), 56k iterations of 25-step episodes; 100
    # evaluations of 100 episodes (EPyMARL's evaluation size). The rule and
    # OptimQ's data come from the variant (LBF_VARIANTS), the OptimQ schedule
    # from _LBF_OPTIMQ.
    "lbf_envs:Foraging-5x5-3p-2f-budget-v0": [
        {"env_args.time_limit": 25, "t_max": 1_400_000, "test_interval": 14_000,
         "test_nepisode": 100},
        {"budget.enabled": True, "budget.initial_budget": [0.55, 0.8, 0.8],
         "budget.observe_budget": "all"},
    ],
    # "lbforaging:Foraging-5x5-2p-1f-coop-pen-v3": [
    #     {"env_args.time_limit": 25, "t_max": 14_000_000, "test_interval": 140_000,
    #      "test_greedy": False, "save_model": True},
    # ],
}

# Named override sets applied last, a fourth grid dimension next to (alg, env,
# seed): every variant runs for every alg and env. Runs are matched to a variant
# by their config, so each variant must set every key in which it differs from
# the others (e.g. "base" sets the exploration policy to None explicitly).
# Use {"default": []} for a sweep without variants.
#
# OptimQ's data (budget.concession_args.optimq.data, components/
# concession_training.py): the warmup fills OptimQ's 100k-transition buffer
# with random play before training; after that, exploration episodes are
# played at a rate per agent episode annealed from 1/4 to 1/16 over 200k
# steps, so about 20% of OptimQ's data is exploration at first and 6% later.
# It never reaches zero, as the FIFO buffer loses the warmup data within ~100k
# agent steps. policy_fraction is set explicitly so these runs are told apart
# from _EXPLORE_ONLY's.
_EXPLORATION = {f"{_OPTIMQ_DATA}explore_policy": "random",
                f"{_OPTIMQ_DATA}warmup_steps": 100_000,
                f"{_OPTIMQ_DATA}policy_fraction": 1.0}
_EXPLORE_ROLLOUTS = {f"{_OPTIMQ_DATA}explore": "anneal",
                     f"{_OPTIMQ_DATA}explore_args.anneal.start": 0.25,
                     f"{_OPTIMQ_DATA}explore_args.anneal.finish": 0.0625,
                     f"{_OPTIMQ_DATA}explore_args.anneal.anneal_time": 200_000,
                     f"{_OPTIMQ_DATA}explore_args.anneal.decay": "linear"}
# 0.16 steps per transition was 4 steps per 25-step episode rollout; PAC's
# parallel rollouts got a tenth of that before, now the same.
_STABLE =  {"budget.concession_args.optimq.training.replay_ratio": 0.16,
           "budget.concession_args.optimq.target_tau": 0.005,
           "budget.concession_args.optimq.loss": "huber",
           "budget.concession_args.optimq.layer_norm": True,
           "budget.concession_args.optimq.reward_range": [-8.0, 0.0]}
# OptimQ trained only on exploration (random-play) episodes, never on the
# agents' own, so the policy cannot shape the data Q* is fitted to. One
# exploration episode per agent episode keeps OptimQ's data (and so its
# gradient steps) the same as when it trained on the agents' episodes.
_EXPLORE_ONLY = {f"{_OPTIMQ_DATA}policy_fraction": 0.0,
                 f"{_OPTIMQ_DATA}explore": "constant",
                 f"{_OPTIMQ_DATA}explore_args.constant.rate": 1.0}
_MATRIX_EXPLORATION = {f"{_OPTIMQ_DATA}explore_policy": "random",
                       f"{_OPTIMQ_DATA}warmup_steps": 5_000,
                       f"{_OPTIMQ_DATA}policy_fraction": 1.0,
                       f"{_OPTIMQ_DATA}explore": "anneal",
                       f"{_OPTIMQ_DATA}explore_args.anneal.start": 0.25,
                       f"{_OPTIMQ_DATA}explore_args.anneal.finish": 0.0625,
                       f"{_OPTIMQ_DATA}explore_args.anneal.anneal_time": 50_000,
                       f"{_OPTIMQ_DATA}explore_args.anneal.decay": "linear"}
# OptimQ's data on LBF: 4 new episodes per agent episode in both, so they
# differ only in where the data comes from. "own": the agents' episodes + 3
# exploration episodes each. "explore_only": 4 exploration episodes each, no
# policy data.
_LBF_OWN = {f"{_OPTIMQ_DATA}policy_fraction": 1.0,
            f"{_OPTIMQ_DATA}explore": "constant",
            f"{_OPTIMQ_DATA}explore_args.constant.rate": 3.0}
_LBF_EXPLORE_ONLY = {f"{_OPTIMQ_DATA}policy_fraction": 0.0,
                     f"{_OPTIMQ_DATA}explore": "constant",
                     f"{_OPTIMQ_DATA}explore_args.constant.rate": 4.0}
VARIANTS = {
    # "base": [{f"{_OPTIMQ_DATA}explore_policy": None}],
    # "warmup": [_EXPLORATION, {f"{_OPTIMQ_DATA}explore_args.constant.rate": 0.0}],
    # "warmup_explore": [_EXPLORATION, _EXPLORE_ROLLOUTS],
    # OptimQ stability pass (components/optimq.py), no exploration. "stable" is
    # every fix together: 0.16 critic steps per transition (from 0.8, ~8 samples
    # per transition as in DQN), a slower target net, Huber loss, LayerNorm and
    # bootstrapped values clamped to the Q* range. SelfishSpread rewards are
    # -distance - collisions, so r_max = 0; r_min = -8 is loose (random play
    # reaches -4.0). "stable_noclip" drops the clamp, to see whether the rest
    # holds without it; "slow" keeps only the step and target-net changes.
    # "stable_warmup_explore" is every fix plus warmup_explore's random-play
    # warmup and exploration rollouts (_EXPLORATION overrides _STABLE's policy).
    # "stable": [_STABLE],
    # "stable_warmup_explore": [_STABLE, _EXPLORATION, _EXPLORE_ROLLOUTS],
    # stable_warmup_explore with OptimQ trained only on exploration (_EXPLORE_ONLY).
    "stable_explore_only":[_STABLE, _EXPLORATION, _EXPLORE_ROLLOUTS, _EXPLORE_ONLY],
    # stable_warmup_explore with joint_overspend_potential: the same episode
    # return as joint_overspend_increment, paid step by step instead of mostly
    # at the end. Compared against the joint_overspend_increment runs of
    # stable_warmup_explore (selfish_spread_collision.ipynb, section 2), so any
    # difference comes from the denser reward alone. One variant per budget,
    # since the env sets [1, 50, 50].
    "potential": [_STABLE, _EXPLORATION, _EXPLORE_ROLLOUTS,
                  {"budget.rule": "joint_overspend_potential"}],
    "potential_c166": [_STABLE, _EXPLORATION, _EXPLORE_ROLLOUTS,
                       {"budget.rule": "joint_overspend_potential",
                        "budget.initial_budget": [1.0, 6.0, 6.0]}],
    # The potential rule with OptimQ trained only on exploration: the second
    # column of a 2x2 with potential / potential_c166 (same reward objective,
    # only the critic's data differs), testing whether keeping the policy away
    # from Q*'s data stops it steering agent 0's measured concession.
    "potential_explore_only": [_STABLE, _EXPLORATION, _EXPLORE_ROLLOUTS, _EXPLORE_ONLY,
                               {"budget.rule": "joint_overspend_potential"}],
    "potential_explore_only_c166": [_STABLE, _EXPLORATION, _EXPLORE_ROLLOUTS, _EXPLORE_ONLY,
                                    {"budget.rule": "joint_overspend_potential",
                                     "budget.initial_budget": [1.0, 6.0, 6.0]}],
    # "potential" scaled to the 50k-step one-step matrix games (budget-table1-3p-v0,
    # run with --envs budget-table1-3p-v0 --variants matrix_potential): a 5k-step
    # warmup (~185 random samples per joint action of the 27), then one random
    # episode per agent episode and 4 critic steps per transition (as before:
    # 4 per episode-runner rollout), and Table 1's reward range. With one step left the
    # bounds are 0, so the clamp also zeroes the bootstrap past the last step.
    "matrix_potential": [_STABLE, _MATRIX_EXPLORATION, _EXPLORE_ONLY,
                         {"budget.concession_args.optimq.reward_range": [0.0, 16.0],
                          "budget.concession_args.optimq.training.replay_ratio": 4.0,
                          "budget.rule": "joint_overspend_potential"}],
    # LBF (_LBF_OPTIMQ): every budget rule, with OptimQ trained on the agents'
    # episodes plus exploration ("own"), or on exploration only ("explore_only"). Named
    # lbf_<rule>_<data>; built in LBF_VARIANTS below. No _STABLE here: its
    # reward_range would override LBF's.
    **{f"lbf_{tag}_{data}": [{"budget.rule": rule}, settings]
       for tag, rule in [("increment", "joint_overspend_increment"),
                         ("signed", "joint_budget_signed_increment"),
                         ("potential", "joint_overspend_potential")]
       for data, settings in [("own", _LBF_OWN), ("explore_only", _LBF_EXPLORE_ONLY)]},
    # "stable_noclip": [_STABLE, {"budget.concession_args.optimq.reward_range": None}],
    # "slow": [_STABLE, {"budget.concession_args.optimq.loss": "mse",
    #                    "budget.concession_args.optimq.layer_norm": False,
    #                    "budget.concession_args.optimq.reward_range": None}],
}

def with_concession(groups, name):
    """A variant's override groups with OptimQ swapped for another learned
    concession model with the same options (e.g. "optimq_additive_linear"):
    budget.concession set to it and the optimq options moved to its block."""
    prefix = "budget.concession_args.optimq."
    moved = [{(f"budget.concession_args.{name}." + k[len(prefix):] if k.startswith(prefix) else k): v
              for k, v in group.items()} for group in groups]
    return moved + [{"budget.concession": name}]


# matrix_potential with the critic taking the joint action as an input
# (components/optimq_additive.py); every other setting is the same.
VARIANTS["matrix_potential_additive"] = with_concession(VARIANTS["matrix_potential"],
                                                        "optimq_additive_linear")

LBF_VARIANTS = [v for v in VARIANTS if v.startswith("lbf_")]

# The variants run when none are named (--variants picks any in VARIANTS).
SWEEP_VARIANTS = LBF_VARIANTS  # was ["matrix_potential"] for budget-table1

# Mirrored from test_interval unless a dict above sets them explicitly.
LOG_INTERVAL_KEYS = ["log_interval", "runner_log_interval", "learner_log_interval"]


def sacred_name(alg):
    """Sacred names its results dir after the yaml's experiment name, which for the
    PAC configs differs from the --config name (pac_ns -> pac_sarsa_ns)."""
    text = (EPYMARL / "src" / "config" / "algs" / f"{alg}.yaml").read_text()
    return re.search(r'^name:\s*"?([^"\s]+)"?', text, re.M).group(1)


def alg_runner(alg):
    """The runner the alg's yaml uses (default.yaml's "episode" if it sets none)."""
    text = (EPYMARL / "src" / "config" / "algs" / f"{alg}.yaml").read_text()
    match = re.search(r'^runner:\s*"?([^"\s#]+)"?', text, re.M)
    return match.group(1) if match else "episode"


def _budget_rule_is_common(rule):
    """Whether a budget rule pays a common reward (envs/budget/rules.py), loaded
    from its file so the sweep does not import epymarl's env packages."""
    import importlib.util
    path = EPYMARL / "src" / "envs" / "budget" / "rules.py"
    spec = importlib.util.spec_from_file_location("budget_rules", path)
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    return rules.get_rule(rule).common


def launch_settings(alg, merged):
    """(--env-config name, extra sacred args) that follow from a run's settings.
    Budget runs use the budget runner wrapping the alg's own runner, the budget
    env config, and the budget rule's common_reward. Kept out of overrides(), so
    completed runs are still matched on the settings above alone."""
    if not merged.get("budget.enabled"):
        return "gymma", {}
    rule = merged.get("budget.rule", "joint_overspend_increment")  # gymma_budget.yaml's default
    return "gymma_budget", {"runner": f"budget_{alg_runner(alg)}",
                            "common_reward": _budget_rule_is_common(rule)}


def overrides(alg, env, variant, seed):
    """{sacred arg: value} for one run, merging base, alg, env and variant overrides."""
    merged = {}
    alg_overrides = (_MATRIX_ALG_OVERRIDES if ":" not in env
                     else _LBF_ALG_OVERRIDES if env.startswith(LBF_PREFIXES)
                     else ALG_OVERRIDES)
    for group in (BASE_OVERRIDES + alg_overrides.get(alg, [])
                  + ENV_OVERRIDES.get(env, []) + VARIANTS[variant]):
        merged.update(group)
    if env.startswith(LBF_PREFIXES):
        for key in _SCALED_STEP_KEYS:
            if key in merged:
                merged[key] *= T_MAX_SCALE.get(alg, 1)
    merged["env_args.key"] = env if ":" in env else f"matrix_envs:{env}"
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
    """True if a sacred config was trained with all of merged's settings.

    A key missing from the config (an option added after the run was made)
    matches only an off value, False or None, so asking for an option's off
    setting still matches runs that predate it."""
    missing = object()
    for key, value in merged.items():
        node = config
        for part in key.split("."):
            node = node.get(part, missing) if isinstance(node, dict) else missing
            if node is missing:
                break
        if node is missing:
            if value is False or value is None:
                continue
            return False
        if node != value and str(node) != str(value):
            return False
    return True


def completed_runs(algs=ALGS, envs=ENVS, variants=tuple(SWEEP_VARIANTS), seeds=SEEDS):
    """(alg, env, variant, seed) -> run_dir for runs already trained with these
    overrides.

    Later run numbers win if a combination was trained twice."""
    wanted = {(a, e, v, s): overrides(a, e, v, s)
              for a in algs for e in envs for v in variants for s in seeds}
    done = {}
    for alg, env, seed, config, run_dir in _completed(algs):
        for variant in variants:
            run = (alg, env, variant, seed)
            if run in wanted and _matches(config, wanted[run]):
                done[run] = run_dir
    return done


def run_one(alg, env, variant, seed):
    """Train one (alg, env, variant, seed); returns the subprocess return code."""
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{alg}__{env.replace(':', '-')}__{variant}__s{seed}.log"
    merged = overrides(alg, env, variant, seed)
    env_config, launch = launch_settings(alg, merged)
    args = [f"{k}={v}" for k, v in {**merged, **launch}.items()]
    cmd = [sys.executable, "src/main.py", f"--config={alg}", f"--env-config={env_config}",
           "with"] + args
    proc_env = os.environ | {"PYTHONPATH": str(PROJECT), "OMP_NUM_THREADS": "1"}
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=EPYMARL, env=proc_env,
                              stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def run_sweep(algs=ALGS, envs=ENVS, variants=tuple(SWEEP_VARIANTS), seeds=SEEDS,
              max_workers=None):
    """Train every missing (alg, env, variant, seed) in the grid; skip what's
    already done.

    Returns the list of (alg, env, variant, seed) runs that failed (empty if all ok)."""
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown} (available: {list(VARIANTS)})")
    max_workers = max_workers or max(1, (os.cpu_count() or 2) // 2)
    done = completed_runs(algs, envs, variants, seeds)
    todo = [(a, e, v, s) for a in algs for e in envs for v in variants for s in seeds
            if (a, e, v, s) not in done]
    print(f"{len(done)} runs already complete, {len(todo)} to train "
          f"({max_workers} workers)", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_one, *run): run for run in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            run, rc = futures[fut], fut.result()
            print(f"[{i}/{len(todo)}] {run[0]} / {run[1]} / {run[2]} / seed {run[3]}: "
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
    parser.add_argument("--variants", type=str, default=None,
                        help=f"comma-separated variants (default: {','.join(SWEEP_VARIANTS)}; available: {','.join(VARIANTS)})")
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
    variants = args.variants.split(",") if args.variants else list(SWEEP_VARIANTS)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS

    failures = run_sweep(algs=algs, envs=envs, variants=variants, seeds=seeds,
                         max_workers=args.max_workers)
    if failures:
        print(f"{len(failures)} runs failed (logs in {LOGS}): {failures}",
              file=sys.stderr)
        return 1
    print("sweep complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
