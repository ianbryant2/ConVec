"""Trains a learned concession model (OptimQ) on its own data and schedule.

Settings live in budget.concession_args.<name>.data / .training:

  data.policy_fraction   share of the agents' training episodes it trains on
  data.explore_policy    policy for its own exploration episodes (null: none),
                         "random" or "greedy"; options in data.explore_policy_args
  data.warmup_steps      exploration env steps before the agents start
  data.explore           exploration episodes per agent episode, over t_env
  training.replay_ratio  gradient steps per critic per new transition

Amounts are per agent episode, so a parallel runner gives the model
batch_size_run times the data. Fractions carry over between rollouts.
Exploration episodes never reach the agents' buffer, t_env or episode stats.
"""
import torch as th

from components.epsilon_schedules import DecayThenFlatSchedule


class RandomExploration:
    """Each agent picks uniformly among its available actions, independently."""

    def __init__(self, concession):
        pass

    def select_actions(self, mac, batch, t_ep, t_env, bs, avail_actions):
        avail = avail_actions.float()
        actions = th.multinomial(avail.reshape(-1, avail.shape[-1]), 1)
        return actions.view(avail.shape[:-1])


class GreedyExploration:
    """Epsilon-greedy on one agent's critic head, round robin over the agents.

    Exploration episodes take the agents' heads in turn (0, 1, ..., n-1, 0, ...)
    and play the joint action maximising that agent's Q* (OptimQ.greedy_actions);
    each agent's action is then independently replaced by a uniformly random
    available one with probability epsilon(t_env), one of EXPLORE_SCHEDULES
    (so warmup, at t_env 0, plays epsilon's start value).
    """

    def __init__(self, concession, epsilon, epsilon_args):
        if not hasattr(concession.model, "greedy_actions"):
            raise ValueError("explore_policy 'greedy' needs a learned Q* concession model (optimq*)")
        self.concession = concession
        self.epsilon = _registered(EXPLORE_SCHEDULES, epsilon, "epsilon schedule")(
            **(epsilon_args.get(epsilon) or {})
        )
        self._next_head = 0
        self._heads = None  # (B,) head of each env in the current rollout

    def select_actions(self, mac, batch, t_ep, t_env, bs, avail_actions):
        if t_ep == 0:
            n = batch.batch_size
            self._heads = (self._next_head + th.arange(n)) % self.concession.n_agents
            self._next_head = (self._next_head + n) % self.concession.n_agents
        states = self.concession._without_budget(batch["state"][bs, t_ep]).float()
        return self.concession.model.greedy_actions(
            states, avail_actions, t_ep, self._heads[bs], self.epsilon(t_env)
        )


class ConstantExplore:
    """The same exploration rate throughout."""

    def __init__(self, rate):
        self.rate = rate

    def __call__(self, t_env):
        return self.rate


class AnnealExplore:
    """Decays from start to finish over anneal_time env steps, then stays."""

    def __init__(self, start, finish, anneal_time, decay):
        self.schedule = DecayThenFlatSchedule(start, finish, anneal_time, decay=decay)

    def __call__(self, t_env):
        return self.schedule.eval(t_env)


# cls(**data.explore_args[name]), called as schedule(t_env) -> episodes per agent episode.
EXPLORE_SCHEDULES = {
    "constant": ConstantExplore,
    "anneal": AnnealExplore,
}


# cls(concession, **data.explore_policy_args[name]) with
# select_actions(mac, batch, t_ep, t_env, bs, avail_actions) -> (B, n_agents).
EXPLORATION_POLICIES = {
    "random": RandomExploration,
    "greedy": GreedyExploration,
}


def _registered(registry, name, what):
    if name not in registry:
        raise ValueError(f"Unknown {what} '{name}' (available: {sorted(registry)})")
    return registry[name]


class ConcessionTrainer:
    """Feeds the runner's RunnerConcession its data and gradient steps."""

    def __init__(self, concession, args, logger):
        data, training = concession.data, concession.training
        self.concession = concession
        self.policy_fraction = data["policy_fraction"]
        if not 0 <= self.policy_fraction <= 1:
            raise ValueError(f"data.policy_fraction must be in [0, 1], got {self.policy_fraction}")
        name = data["explore_policy"]
        self.explorer = (
            None if name is None
            else _registered(EXPLORATION_POLICIES, name, "exploration policy")(
                concession, **((data.get("explore_policy_args") or {}).get(name) or {})
            )
        )
        self.warmup_steps = data["warmup_steps"]
        explore = data["explore"]
        self.explore_rate = _registered(EXPLORE_SCHEDULES, explore, "explore schedule")(
            **(data["explore_args"].get(explore) or {})
        )
        if self.explorer is None and (self.warmup_steps > 0 or self.explore_rate(0) > 0):
            raise ValueError("data.warmup_steps and data.explore need data.explore_policy set")
        if self.explorer is None and self.policy_fraction == 0:
            raise ValueError(
                "data.policy_fraction 0 without data.explore_policy: the concession "
                "model would never get any data"
            )
        self.replay_ratio = training["replay_ratio"]

        self.logger = logger
        self.log_interval = args.learner_log_interval
        self.last_log_t = -self.log_interval - 1
        self.explore_rollouts = 0
        self.explored_episodes = 0
        # Owed but not yet taken: fractions of an episode or a gradient step.
        self._policy_credit = 0.0
        self._explore_credit = 0.0
        self._step_credit = 0.0
        # Exploration episodes played but not used yet: (batch, records, first unused).
        self._spare = None
        # True once the model gets no more data, and so no more updates.
        self.frozen = False

    def warmup(self, runner):
        """Explore and train for warmup_steps env steps before the agents start."""
        while runner.t_explore < self.warmup_steps:
            self._train(self._explore(runner, runner.batch_size))
        if self.warmup_steps:
            self.logger.console_logger.info(
                f"Concession warmup: {self.explored_episodes} exploration episodes, "
                f"{runner.t_explore} env steps"
            )

    def observe(self, runner, batch, records):
        """After each training rollout: add its policy share and owed
        exploration episodes, then train."""
        self.concession.record_deltas(batch, records)
        n = batch.batch_size
        data = []
        n_policy = self._take("_policy_credit", self.policy_fraction * n)
        if n_policy:
            data.append(self.concession.transitions(batch, records, episodes=slice(0, n_policy)))
        rate = self.explore_rate(runner.t_env) if self.explorer is not None else 0.0
        data += self._explore(runner, self._take("_explore_credit", rate * n))
        self._train(data)
        self.concession.log(runner.t_env, runner.logger)
        # No policy share and exploration annealed to 0 (the schedules only
        # decay): the model is final, so save it once.
        if not self.frozen and self.policy_fraction == 0 and rate == 0:
            self.frozen = True
            runner.save_concession("frozen")

        if runner.t_env - self.last_log_t >= self.log_interval:
            self.logger.log_stat("concession_explore_rate", rate, runner.t_env)
            self.logger.log_stat("concession_explore_rollouts", self.explore_rollouts, runner.t_env)
            self.logger.log_stat("concession_explore_episodes", self.explored_episodes, runner.t_env)
            self.logger.log_stat("concession_explore_steps", runner.t_explore, runner.t_env)
            self.last_log_t = runner.t_env

    def _take(self, credit, amount):
        """Add amount to a credit and take its whole part."""
        total = getattr(self, credit) + amount
        whole = int(total + 1e-9)
        setattr(self, credit, total - whole)
        return whole

    def _explore(self, runner, episodes):
        """Transitions of `episodes` exploration episodes, reusing leftovers
        from earlier rollouts first."""
        data = []
        while episodes > 0:
            if self._spare is None:
                batch, records = runner.explore_rollout(self.explorer)
                self._spare = (batch, records, 0)
                self.explore_rollouts += 1
            batch, records, start = self._spare
            n = min(episodes, batch.batch_size - start)
            data.append(self.concession.transitions(batch, records, episodes=slice(start, start + n)))
            episodes -= n
            self.explored_episodes += n
            start += n
            self._spare = None if start == batch.batch_size else (batch, records, start)
        return data

    def _train(self, data):
        """Add transitions, then take replay_ratio steps per transition."""
        if not data:
            return
        transitions = {k: th.cat([d[k] for d in data]) for k in data[0]}
        self.concession.add(transitions)
        n_steps = self._take("_step_credit", self.replay_ratio * len(transitions["state"]))
        self.concession.fit(n_steps, transitions)
