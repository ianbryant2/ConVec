"""Runner side of the concession budget: computes each step's concession delta.

When budget.enabled is True, the runner calls the concession model after the
agents act and sends (actions, delta) to BudgetWrapper, which spends it. The
env never computes delta itself, so a critic trained by the learner can be used
as it learns. The replay buffer still stores only the actions, so learners see
the ordinary (s, a, r, s') contract with s = (s, b), r = r^c.

A concession model is chosen by budget.concession, built as
``cls(args, state_dim, **budget.concession_args[name])`` and called as

    model(states, actions, avail_actions, t) -> (B, n_agents), undiscounted

with tensors ``states`` (B, state_dim) the joint state before the step, without
the budget part, ``actions`` (B, n_agents) the joint action taken there,
``avail_actions`` (B, n_agents, n_actions) and ``t`` the step index within the
episode. The true delta is >= 0; a learned model's estimate may dip slightly
below it, and is not clamped so its noise cancels over an episode. A model with
an ``update(transitions)`` method is also trained, after each training rollout,
on the transitions (with their step index "t") and the original per-agent
rewards.
"""
from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch as th

from components.optimq import OptimQ


class UtopianTableConcession:
    """Exact delta for single-step matrix_envs games: the best payoff an agent
    gets anywhere in the table, minus the payoff of the joint action taken."""

    def __init__(self, args, state_dim):
        key = args.env_args["key"]
        env = gym.make(key)
        game = getattr(env.unwrapped, "game", None)
        env.close()
        if game is None:
            raise ValueError(
                f"concession 'utopian_table' needs a matrix_envs game, got '{key}'"
            )
        self.payoffs = game.rewards  # (|A_1|, ..., |A_n|, n_agents)
        self.utopian_values = game.rewards.reshape(-1, args.n_agents).max(axis=0)

    def __call__(self, states, actions, avail_actions, t):
        actions = actions.cpu().numpy()
        return self.utopian_values - self.payoffs[tuple(actions.T)]


CONCESSIONS = {
    "utopian_table": UtopianTableConcession,
    "optimq": OptimQ,
}


def add_budget_scheme(scheme, args):
    """Batch fields the budget env reports every step (for the Q* critics)."""
    if args.budget["enabled"]:
        scheme["concession"] = {"vshape": (args.n_agents,)}
        scheme["env_reward"] = {"vshape": (args.n_agents,)}


def pop_budget_step_data(env_info):
    """Remove the budget env's per-step arrays from info (the runners sum numeric
    info values into stats): {batch key: tuple}, empty for other envs."""
    return env_info.pop("budget_step", {})


class RunnerConcession:
    """Computes delta in the runner and pairs it with the joint actions."""

    def __init__(self, args):
        budget = args.budget
        name = budget["concession"]
        if name not in CONCESSIONS:
            raise ValueError(
                f"Unknown concession '{name}' (available: {sorted(CONCESSIONS)})"
            )
        self.n_agents = args.n_agents
        # The batch state ends with the full budget unless it is unobserved;
        # concession models see the state without it.
        self.budget_dims = 0 if budget["observe_budget"] == "none" else args.n_agents
        self.model = CONCESSIONS[name](
            args,
            args.state_shape - self.budget_dims,
            **(budget["concession_args"].get(name) or {}),
        )
        # On matrix games the utopian table is the exact delta, so a learned
        # model's deltas are scored against it.
        self.reference = None
        if name != "utopian_table" and args.env_args.get("key", "").startswith(
            "matrix_envs:"
        ):
            self.reference = UtopianTableConcession(args, None)
        self.log_interval = args.learner_log_interval
        self.last_log_t = -self.log_interval - 1
        self.stats = defaultdict(list)

    @classmethod
    def maybe_create(cls, args):
        return cls(args) if args.budget["enabled"] else None

    def _without_budget(self, states):
        return states[..., : states.shape[-1] - self.budget_dims]

    def env_actions(self, states, actions, avail_actions, t):
        """Batch tensors for B envs at step t -> list of (actions, delta)."""
        actions = actions.reshape(-1, self.n_agents)
        deltas = self.model(self._without_budget(states), actions, avail_actions, t)
        if hasattr(deltas, "cpu"):
            deltas = deltas.cpu().numpy()
        deltas = np.asarray(deltas, dtype=float)
        return [(a, d) for a, d in zip(actions.cpu().numpy(), deltas)]

    def train(self, batch, t_env, logger, record_deltas=True):
        """After a training rollout: record the deltas charged in it, and train a
        learnable concession model on its transitions, with the original
        per-agent rewards the env reported (batch "env_reward"). Exploration
        rollouts pass record_deltas=False, so the logged deltas stay those of
        the agents' own play."""
        # Valid transitions, masked as the learners do.
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        valid = mask.squeeze(-1).bool()
        actions = batch["actions"][:, :-1, :, 0][valid]

        if record_deltas:
            self._record_deltas(batch["concession"][:, :-1][valid], actions)

        if hasattr(self.model, "update"):
            states = self._without_budget(batch["state"])
            steps = th.arange(valid.shape[1], device=valid.device).expand_as(valid)
            stats = self.model.update(
                {
                    "state": states[:, :-1][valid],
                    "t": steps[valid].float(),
                    "actions": actions,
                    "env_reward": batch["env_reward"][:, :-1][valid],
                    "next_state": states[:, 1:][valid],
                    "next_avail": batch["avail_actions"][:, 1:][valid],
                    "terminated": terminated.squeeze(-1)[valid].unsqueeze(-1),
                }
            )
            for k, v in (stats or {}).items():
                self.stats[k].append(v)

        if self.stats and t_env - self.last_log_t >= self.log_interval:
            for k, v in self.stats.items():
                logger.log_stat(k, float(np.mean(v)), t_env)
            self.stats.clear()
            self.last_log_t = t_env

    def _record_deltas(self, deltas, actions):
        """Stats of the (N, n_agents) deltas charged at N valid transitions."""
        if len(deltas) == 0:
            return
        for i in range(self.n_agents):
            self.stats[f"concession_delta_{i}"].append(deltas[:, i].mean().item())
        # Share of charges that are exactly 0 (a best joint action taken, under
        # the exact table), and below 0 (a learned model's estimation noise).
        self.stats["concession_zero_frac"].append((deltas == 0).float().mean().item())
        self.stats["concession_neg_frac"].append((deltas < 0).float().mean().item())
        if self.reference is not None:
            exact = th.as_tensor(
                self.reference(None, actions, None, None),
                dtype=deltas.dtype,
                device=deltas.device,
            )
            err = deltas - exact
            # Positive bias: the model overcharges, and budgets run out early.
            self.stats["concession_bias"].append(err.mean().item())
            self.stats["concession_abs_err"].append(err.abs().mean().item())
