"""Concession models: compute each agent's delta for the budget runners.

A model is chosen by budget.concession, built as
``cls(args, state_dim, **budget.concession_args[name])`` and called as

    model(states, actions, avail_actions, t) -> (B, n_agents) deltas

with states (B, state_dim) excluding the budget. Models with add() and
train() learn; their data/training options go to concession_training.py.

budget.concession_args.<name>.delta picks the charge, made after each step:
  "advantage"  V_i(s_t) - Q_i(s_t, a_t), the model's own delta (default)
  "td"         V_i(s_t) - r_i,t - gamma (1 - done) V_i(s_t+1), with
               V_i(s) = max_a Q_i(s, a) from model.values(states, avail_actions, t)
Both have the same expected value under the exact Q*. td's discounted
episode total telescopes to V_i(s_0) - G_i, so a learned model's errors at
the intermediate states cancel; it adds the env's reward and transition noise.
"""
import copy
from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch as th

from components.optimq import OptimQ
from components.optimq_additive import OptimQAdditiveLinear, OptimQAdditiveMLP
from envs.budget.tracker import budget_state_size


class UtopianTableConcession:
    """Exact delta for one-step matrix games: best payoff minus payoff taken."""

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

    def values(self, states, avail_actions, t):
        # One-step games: the value after the step is masked out by done.
        return np.broadcast_to(self.utopian_values, (len(states), len(self.utopian_values)))


CONCESSIONS = {
    "utopian_table": UtopianTableConcession,
    "optimq": OptimQ,
    "optimq_additive_linear": OptimQAdditiveLinear,
    "optimq_additive_mlp": OptimQAdditiveMLP,
}

DELTAS = ("advantage", "td")


class RunnerConcession:
    """Wraps a concession model for the runner: deltas, training data, stats."""

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
        self.budget_dims = budget_state_size(budget["observe_budget"], args.n_agents)
        options = dict(budget["concession_args"].get(name) or {})
        self.data = options.pop("data", None)
        self.training = options.pop("training", None)
        self.delta = options.pop("delta", "advantage")
        if self.delta not in DELTAS:
            raise ValueError(f"Unknown concession delta '{self.delta}' (available: {list(DELTAS)})")
        self.gamma = args.gamma
        self.model = CONCESSIONS[name](args, args.state_shape - self.budget_dims, **options)
        if self.delta == "td" and not hasattr(self.model, "values"):
            raise ValueError(f"concession '{name}' has no values(), which delta 'td' needs")
        self.learns = hasattr(self.model, "train")
        if self.learns and (self.data is None or self.training is None):
            raise ValueError(
                f"concession '{name}' learns: budget.concession_args.{name} needs "
                "'data' and 'training' blocks"
            )
        # On matrix games the utopian table is the exact delta, so a learned
        # model's deltas are scored against it.
        self.reference = None
        if name != "utopian_table" and args.env_args.get("key", "").startswith(
            "matrix_envs:"
        ):
            self.reference = UtopianTableConcession(args, None)
        self.device = args.device
        self.log_interval = args.learner_log_interval
        self.last_log_t = -self.log_interval - 1
        self.stats = defaultdict(list)

    def _without_budget(self, states):
        return states[..., : states.shape[-1] - self.budget_dims]

    def deltas(self, states, actions, avail_actions, t, rewards, next_states, next_avail, done):
        """(B, n_agents) deltas for B envs' step t, after it: the states and
        avail actions before and after it (states without budget), its
        per-agent env rewards and whether it ended the episode."""

        def tensor(x, **kwargs):
            if not isinstance(x, th.Tensor):
                x = np.asarray(x)
            return th.as_tensor(x, device=self.device, **kwargs)

        def array(x):
            if hasattr(x, "cpu"):
                x = x.cpu().numpy()
            return np.asarray(x, dtype=float)

        states = tensor(states, dtype=th.float32)
        avail_actions = tensor(avail_actions)
        if self.delta == "advantage":
            return array(self.model(
                states, tensor(actions).reshape(-1, self.n_agents), avail_actions, t
            ))
        value = array(self.model.values(states, avail_actions, t))
        next_value = array(self.model.values(
            tensor(next_states, dtype=th.float32), tensor(next_avail), t + 1
        ))
        alive = 1.0 - np.asarray(done, dtype=float).reshape(-1, 1)
        return value - np.asarray(rewards, dtype=float) - self.gamma * alive * next_value

    def _valid(self, batch):
        """Mask of a rollout's valid transitions, as the learners build it."""
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        return mask.squeeze(-1).bool(), terminated

    def transitions(self, batch, records, episodes=None):
        """A rollout's valid transitions with per-agent env rewards, optionally
        only the episodes in the slice `episodes`."""
        if episodes is not None:
            batch = batch[episodes]
            records = {k: v[episodes] for k, v in records.items()}
        valid, terminated = self._valid(batch)
        states = self._without_budget(batch["state"])
        steps = th.arange(valid.shape[1], device=valid.device).expand_as(valid)
        return {
            "state": states[:, :-1][valid],
            "t": steps[valid].float(),
            "actions": batch["actions"][:, :-1, :, 0][valid],
            "env_reward": records["env_reward"][:, : valid.shape[1]][valid],
            "next_state": states[:, 1:][valid],
            "next_avail": batch["avail_actions"][:, 1:][valid],
            "terminated": terminated.squeeze(-1)[valid].unsqueeze(-1),
        }

    def record_deltas(self, batch, records):
        """Stats of the deltas charged in a training rollout."""
        valid, _ = self._valid(batch)
        actions = batch["actions"][:, :-1, :, 0][valid]
        deltas = records["concession"][:, : valid.shape[1]][valid]
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

    def save(self, path):
        """Save the model for scoring later (scripts/lbf_feasibility.py --load),
        without its replay buffers, so it can't be trained further."""
        model = copy.copy(self.model)
        if hasattr(model, "buffers"):
            model.buffers = []
        th.save(model, path)

    def add(self, transitions):
        """Add transitions to a learned model's buffer."""
        self.model.add(transitions)

    def fit(self, n_steps, transitions):
        """Take n_steps gradient steps; stats are measured on transitions."""
        self._keep(self.model.train(n_steps, transitions))

    def _keep(self, stats):
        for k, v in (stats or {}).items():
            self.stats[k].append(v)

    def log(self, t_env, logger):
        if self.stats and t_env - self.last_log_t >= self.log_interval:
            for k, v in self.stats.items():
                logger.log_stat(k, float(np.mean(v)), t_env)
            self.stats.clear()
            self.last_log_t = t_env
