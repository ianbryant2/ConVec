"""Exploration rollouts for a learned concession model (OptimQ).

The Q*_i critics need every joint action tried in the states they are asked
about, which the agents' own rollouts rarely give them. This adds extra rollouts
played by an exploration policy whose transitions train only the concession
model: they never enter the learner's replay buffer, do not advance t_env and
are left out of the logged episode stats. Q*_i does not depend on the policy
that collected its data, so these transitions are as valid for it as the
agents' own.

Two uses of the same exploration policy, set in budget.exploration:
  warmup_rollouts   exploration rollouts before training starts, so delta is
                    sensible from the first learner update
  interval, prob_*  after every `interval` training rollouts, one exploration
                    rollout with probability prob, which decays from prob_start
                    to prob_finish over prob_anneal_time env steps

An exploration policy is chosen by budget.exploration.policy, built as
``cls(args)`` and called as

    policy.select_actions(mac, batch, t_ep, t_env, bs, avail_actions) -> (B, n_agents)

with the arguments of mac.select_actions plus ``avail_actions`` (B, n_agents,
n_actions) for the envs in ``bs``. It may ignore ``mac`` (as "random" does) or
build on the agents' current policy.
"""
import numpy as np
import torch as th

from components.budget_concession import CONCESSIONS
from components.epsilon_schedules import DecayThenFlatSchedule


class RandomExploration:
    """Each agent picks uniformly among its available actions, independently."""

    def __init__(self, args):
        pass

    def select_actions(self, mac, batch, t_ep, t_env, bs, avail_actions):
        avail = avail_actions.float()
        actions = th.multinomial(avail.reshape(-1, avail.shape[-1]), 1)
        return actions.view(avail.shape[:-1])


EXPLORATION_POLICIES = {
    "random": RandomExploration,
}


class ConcessionExploration:
    def __init__(self, args, logger):
        cfg = args.budget["exploration"]
        name = cfg["policy"]
        if name not in EXPLORATION_POLICIES:
            raise ValueError(
                f"Unknown exploration policy '{name}' "
                f"(available: {sorted(EXPLORATION_POLICIES)})"
            )
        self.policy = EXPLORATION_POLICIES[name](args)
        self.warmup_rollouts = cfg["warmup_rollouts"]
        self.interval = cfg["interval"]
        if self.interval < 1:
            raise ValueError(f"budget.exploration.interval must be >= 1, got {self.interval}")
        self.prob = DecayThenFlatSchedule(
            cfg["prob_start"],
            cfg["prob_finish"],
            cfg["prob_anneal_time"],
            decay=cfg["prob_decay"],
        )
        self.logger = logger
        self.log_interval = args.learner_log_interval
        self.last_log_t = -self.log_interval - 1
        self.training_rollouts = 0
        self.rollouts = 0

    @classmethod
    def maybe_create(cls, args, logger):
        """None unless budget.exploration.policy is set. Only a concession model
        that learns (has update()) can use the extra rollouts."""
        budget = args.budget
        if not budget["enabled"] or budget["exploration"]["policy"] is None:
            return None
        if not hasattr(CONCESSIONS[budget["concession"]], "update"):
            raise ValueError(
                f"budget.exploration needs a learned concession model; "
                f"'{budget['concession']}' does not learn"
            )
        return cls(args, logger)

    def warmup(self, runner):
        for _ in range(self.warmup_rollouts):
            self._explore(runner)
        if self.warmup_rollouts:
            self.logger.console_logger.info(
                f"Concession exploration warmup: {self.warmup_rollouts} rollouts, "
                f"{runner.t_explore} env steps"
            )

    def after_training_rollout(self, runner):
        """Call once after each training rollout."""
        self.training_rollouts += 1
        prob = self.prob.eval(runner.t_env)
        if self.training_rollouts % self.interval == 0 and np.random.rand() < prob:
            self._explore(runner)
        if runner.t_env - self.last_log_t >= self.log_interval:
            self.logger.log_stat("concession_explore_prob", prob, runner.t_env)
            self.logger.log_stat("concession_explore_rollouts", self.rollouts, runner.t_env)
            self.logger.log_stat("concession_explore_steps", runner.t_explore, runner.t_env)
            self.last_log_t = runner.t_env

    def _explore(self, runner):
        runner.run(test_mode=False, explorer=self.policy)
        self.rollouts += 1
