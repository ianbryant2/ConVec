"""Budget runners: the stock episode and parallel runners with a concession budget.

Each step, between the agents acting and the env stepping, the runner computes
the concession delta, spends it from that env's budget, replaces the rewards
with the budget rule's and appends the budget to observations and state. The
envs stay stock. After each training rollout a learned concession model is
trained (components/concession_training.py).

Use runner=budget_episode / budget_parallel with --env-config=gymma_budget.
"""
import numpy as np
import torch as th

from components.budget_concession import RunnerConcession
from components.concession_training import ConcessionTrainer
from envs.budget.tracker import (
    EpisodeBudget,
    budget_obs_size,
    budget_state_size,
    matrix_game_reference,
)
from .episode_runner import EpisodeRunner
from .parallel_runner import ParallelRunner


class BudgetRunnerMixin:
    """Budget logic shared by both runners; subclasses implement _wrap_envs()."""

    def __init__(self, args, logger):
        budget = getattr(args, "budget", None)
        if not budget or not budget.get("enabled"):
            raise ValueError(
                f"runner '{args.runner}' needs budget.enabled=True "
                "(and --env-config=gymma_budget for the budget settings)"
            )
        if args.env != "gymma":
            raise ValueError(f"the budget runners need env 'gymma', got '{args.env}'")
        # The budget rule decides the learners' reward; the env keeps its
        # per-agent rewards, which the concession model trains on.
        common_reward = args.common_reward
        args.common_reward = False
        try:
            super().__init__(args, logger)
        finally:
            args.common_reward = common_reward

        n_agents = super().get_env_info()["n_agents"]
        self.budgets = [
            EpisodeBudget(
                n_agents,
                common_reward,
                budget["rule"],
                budget["initial_budget"],
                args.gamma,
                budget["observe_budget"],
            )
            for _ in range(self.batch_size)
        ]
        matrix = matrix_game_reference(
            args.env_args.get("key"), self.budgets[0].initial_budget, n_agents
        )
        for b in self.budgets:
            b._matrix = matrix

        # Env steps of exploration rollouts, kept out of t_env.
        self.t_explore = 0
        # Built in setup(), once args.state_shape is set.
        self.concession = None
        self.trainer = None  # only for a learned concession model
        self._warmed_up = False
        self._wrap_envs()

    def setup(self, scheme, groups, preprocess, mac):
        super().setup(scheme, groups, preprocess, mac)
        self.concession = RunnerConcession(self.args)
        if self.concession.learns:
            self.trainer = ConcessionTrainer(self.concession, self.args, self.logger)

    def get_env_info(self):
        info = dict(super().get_env_info())
        observe = self.args.budget["observe_budget"]
        info["obs_shape"] += budget_obs_size(observe, info["n_agents"])
        info["state_shape"] += budget_state_size(observe, info["n_agents"])
        return info

    def run(self, test_mode=False):
        if test_mode:
            return super().run(test_mode=True)
        if not self._warmed_up:
            self._warmed_up = True
            if self.trainer is not None:
                self.trainer.warmup(self)

        batch = super().run(test_mode=False)
        records = self._records(batch)
        if self.trainer is not None:
            self.trainer.observe(self, batch, records)
        else:
            self.concession.record_deltas(batch, records)
            self.concession.log(self.t_env, self.logger)
        return batch

    def explore_rollout(self, explorer):
        """One rollout played by explorer, kept out of t_env and the stats.
        Returns (batch, records)."""
        t_env = self.t_env
        log_train_stats_t = self.log_train_stats_t
        n_returns = len(self.train_returns)
        train_stats = dict(self.train_stats)
        mac = self.mac
        self.mac = _ExplorerMAC(explorer, mac)
        self.log_train_stats_t = float("inf")  # no runner logging during the rollout
        try:
            batch = super().run(test_mode=False)
        finally:
            self.mac = mac
            self.t_explore += self.t_env - t_env
            self.t_env = t_env
            self.log_train_stats_t = log_train_stats_t
            del self.train_returns[n_returns:]
            self.train_stats.clear()
            self.train_stats.update(train_stats)
        return batch, self._records(batch)

    def _records(self, batch):
        """The last rollout's env rewards and deltas, each (B, T, n_agents)."""
        B, T = batch.batch_size, batch.max_seq_length
        n_agents = self.budgets[0].n_agents
        records = {k: th.zeros(B, T, n_agents) for k in ("env_reward", "concession")}
        for i, budget in enumerate(self.budgets[:B]):
            n = len(budget.env_rewards)
            if n:
                records["env_reward"][i, :n] = th.tensor(budget.env_rewards)
                records["concession"][i, :n] = th.tensor(budget.concessions)
        return {k: v.to(batch.device) for k, v in records.items()}


class BudgetEpisodeRunner(BudgetRunnerMixin, EpisodeRunner):
    def _wrap_envs(self):
        self.env = _BudgetEnv(self.env, self.budgets[0], self)


class BudgetParallelRunner(BudgetRunnerMixin, ParallelRunner):
    def _wrap_envs(self):
        self.parent_conns = _BudgetPipes(self.parent_conns, self.budgets, self).conns


class _ExplorerMAC:
    """Stands in for the agents' controller during an exploration rollout."""

    def __init__(self, explorer, mac):
        self.explorer = explorer
        self.mac = mac

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        return self.explorer.select_actions(
            self.mac,
            ep_batch,
            t_ep=t_ep,
            t_env=t_env,
            bs=bs,
            avail_actions=ep_batch["avail_actions"][bs, t_ep],
        )

    def __getattr__(self, name):
        return getattr(self.mac, name)


class _BudgetEnv:
    """The episode runner's env (in this process), with the budget applied."""

    def __init__(self, env, budget, runner):
        self._env = env
        self._budget = budget
        self._runner = runner

    def reset(self, *args, **kwargs):
        out = self._env.reset(*args, **kwargs)
        self._budget.reset()
        return out

    def get_obs(self):
        return self._budget.observe(self._env.get_obs())

    def get_state(self):
        return self._budget.state(self._env.get_state())

    def step(self, actions):
        delta = self._runner.concession.deltas(
            [self._env.get_state()],
            actions,
            [self._env.get_avail_actions()],
            self._budget.t,
        )[0]
        _, env_rewards, done, truncated, info = self._env.step(actions)
        reward, budget_info = self._budget.step(actions, delta, env_rewards, done or truncated)
        return self.get_obs(), reward, done, truncated, {**info, **budget_info}

    def __getattr__(self, name):
        return getattr(self._env, name)


class _BudgetPipes:
    """The parallel runner's env pipes, with the budget applied. Step messages
    are held until a reply is needed, so all envs' deltas come from one
    batched model call."""

    def __init__(self, conns, budgets, runner):
        self._raw = list(conns)
        self._budgets = budgets
        self._runner = runner
        n = len(self._raw)
        self._pending = {}  # env index -> actions, step not sent yet
        self._last_cmd = [None] * n
        self._state = [None] * n  # each env's latest state and avail actions, without the budget
        self._avail = [None] * n
        self._actions = [None] * n  # the last step sent: its actions and deltas
        self._delta = [None] * n
        self.conns = [_BudgetPipe(self, i) for i in range(n)]

    def send(self, i, msg):
        cmd, data = msg
        if cmd == "step":
            self._pending[i] = data
            return
        self._flush()
        if cmd == "reset":
            self._budgets[i].reset()
        self._last_cmd[i] = cmd
        self._raw[i].send(msg)

    def recv(self, i):
        self._flush()
        data = self._raw[i].recv()
        cmd = self._last_cmd[i]
        if cmd not in ("reset", "step"):
            return data
        budget = self._budgets[i]
        data = dict(data)
        if cmd == "step":
            reward, budget_info = budget.step(
                self._actions[i], self._delta[i], data["reward"], data["terminated"]
            )
            data["reward"] = reward
            data["info"] = {**data["info"], **budget_info}
        self._state[i] = data["state"]
        self._avail[i] = data["avail_actions"]
        data["obs"] = budget.observe(data["obs"])
        data["state"] = budget.state(data["state"])
        return data

    def _flush(self):
        if not self._pending:
            return
        envs = sorted(self._pending)
        t = self._budgets[envs[0]].t
        assert all(self._budgets[i].t == t for i in envs), "envs out of step"
        actions = np.stack([self._pending[i] for i in envs])
        deltas = self._runner.concession.deltas(
            [self._state[i] for i in envs], actions, [self._avail[i] for i in envs], t
        )
        for i, delta in zip(envs, deltas):
            self._actions[i] = self._pending[i]
            self._delta[i] = delta
            self._last_cmd[i] = "step"
            self._raw[i].send(("step", self._pending[i]))
        self._pending.clear()


class _BudgetPipe:
    """One env's pipe, as the parallel runner uses it."""

    def __init__(self, pipes, i):
        self._pipes = pipes
        self._i = i

    def send(self, msg):
        self._pipes.send(self._i, msg)

    def recv(self):
        return self._pipes.recv(self._i)

    def __getattr__(self, name):
        return getattr(self._pipes._raw[self._i], name)
