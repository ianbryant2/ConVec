import numpy as np

from ..multiagentenv import MultiAgentEnv
from .rules import BudgetStep, get_rule


class BudgetWrapper(MultiAgentEnv):
    """Adds a per-agent concession budget on top of any MultiAgentEnv.

    step() takes (actions, concession): the runner computes each step's
    concession delta (components/budget_concession.py) and the env only spends
    it. delta is taken off each agent's budget, discounted by gamma^t, and the
    budget rule turns the budget state into the rewards the learners see. The
    inner env must report per-agent rewards (build it with common_reward=False);
    the rule decides the reward shape this env returns.

    Enabled with budget.enabled=True (see default.yaml); envs/__init__.py builds
    the wrapper inside each env process.
    """

    def __init__(self, env, common_reward, rule, initial_budget, gamma, observe_budget):
        self._env = env
        self.n_agents = env.n_agents
        self.episode_limit = env.episode_limit
        self._obs = None

        self.rule_name = rule
        self.rule = get_rule(rule)
        if self.rule.common != common_reward:
            raise ValueError(
                f"budget rule '{rule}' produces "
                f"{'common' if self.rule.common else 'individual'} rewards but "
                f"common_reward={common_reward}; drop the common_reward override "
                "and main.py sets it from the rule"
            )
        self.common_reward = common_reward

        if initial_budget is None:
            raise ValueError(
                "set budget.initial_budget (one number for every agent, or a list)"
            )
        self.initial_budget = np.asarray(initial_budget, dtype=float)
        if self.initial_budget.ndim == 0:
            # One number: every agent gets the same budget, whatever n_agents is.
            self.initial_budget = np.full(self.n_agents, float(self.initial_budget))
        if self.initial_budget.shape != (self.n_agents,):
            raise ValueError(
                f"initial_budget has {self.initial_budget.size} entries, "
                f"env has {self.n_agents} agents"
            )
        self.gamma = gamma
        # What each agent sees of the budget: nothing, its own b_i, or the full
        # vector b (Method A conditions the policy on b).
        if observe_budget not in ("none", "own", "all"):
            raise ValueError(
                f"observe_budget must be 'none', 'own' or 'all', got '{observe_budget}'"
            )
        self.observe_budget = observe_budget
        self._matrix = self._matrix_game_reference()
        self._reset_budget()

    def _reset_budget(self):
        self.budget = self.initial_budget.copy()
        self._t = 0
        # Per-episode totals, reported in info when the episode ends.
        self._env_return = np.zeros(self.n_agents)
        self._env_disc_return = np.zeros(self.n_agents)
        self._first_violation_t = None

    def _matrix_game_reference(self):
        """For single-step matrix_envs games: which joint actions stay within
        the initial budget under exact (utopian) concessions, and the best
        welfare among them. None for other envs."""
        inner = getattr(getattr(self._env, "_env", None), "unwrapped", None)
        game = getattr(inner, "game", None)
        if game is None:
            return None
        payoffs = game.rewards  # (|A_1|, ..., |A_n|, n_agents), expected payoffs
        utopian = payoffs.reshape(-1, self.n_agents).max(axis=0)
        feasible = (utopian - payoffs <= self.initial_budget).all(-1)
        welfare = payoffs.sum(-1)
        best = welfare[feasible].max() if feasible.any() else None
        return {"feasible": feasible, "welfare": welfare, "best": best}

    def _matrix_step_stats(self, actions):
        """Scalars scoring a matrix-game joint action against the constrained
        optimum (logged as <key>_mean)."""
        joint = tuple(actions)
        feasible = bool(self._matrix["feasible"][joint])
        stats = {"feasible": float(feasible)}
        best = self._matrix["best"]
        if best is not None:
            welfare = self._matrix["welfare"][joint]
            stats["constrained_optimal"] = float(feasible and welfare >= best - 1e-9)
            # Negative when an over-budget cell beats the best feasible welfare.
            stats["welfare_regret"] = float(best - welfare)
        return stats

    def _episode_stats(self):
        """Scalars for the end of an episode; the runner logs each as <key>_mean
        over episodes, so 0/1 values become rates."""
        stats = {}
        for i in range(self.n_agents):
            # E[b_i,T] >= 0 certifies pi in C.
            stats[f"budget_{i}_final"] = self.budget[i]
            # Discounted concession charged this episode.
            stats[f"budget_{i}_spent"] = self.initial_budget[i] - self.budget[i]
            # Overspent at the end of the episode (with learned deltas a budget
            # can dip below 0 and recover, so this is the final budget's sign).
            stats[f"violated_{i}"] = self.budget[i] < 0
            stats[f"env_return_{i}"] = self._env_return[i]
            stats[f"env_disc_return_{i}"] = self._env_disc_return[i]
        stats["env_return_total"] = self._env_return.sum()
        stats["violated_any"] = (self.budget < 0).any()
        # Steps taken before any budget went negative (all of them if none did).
        stats["steps_within_budget"] = (
            self._t if self._first_violation_t is None else self._first_violation_t
        )
        return {k: float(v) for k, v in stats.items()}

    def _budget_obs_size(self):
        return {"none": 0, "own": 1, "all": self.n_agents}[self.observe_budget]

    def _observe(self):
        """The inner env's observations with the budget each agent can see."""
        obs = self._env.get_obs()
        if self.observe_budget == "none":
            return obs
        if self.observe_budget == "own":
            return [np.append(o, self.budget[i]) for i, o in enumerate(obs)]
        return [np.concatenate([o, self.budget]) for o in obs]

    def _split_action(self, action):
        """Return (joint action, concession) from what step() received."""
        if not (isinstance(action, tuple) and len(action) == 2):
            raise ValueError(
                "BudgetWrapper.step expects (actions, concession) from the runner"
            )
        actions, concession = action
        concession = np.asarray(concession, dtype=float)
        if concession.shape != (self.n_agents,):
            raise ValueError(
                f"concession has shape {concession.shape}, expected ({self.n_agents},)"
            )
        # The true delta_i = V*_i - Q*_i is >= 0 (Lemma 14 of the writeup), but a
        # learned estimate may dip below 0; it is spent unclamped so that its
        # noise cancels over an episode instead of adding up.
        if not np.all(np.isfinite(concession)):
            raise ValueError(f"concession must be finite, got {concession}")
        return [int(a) for a in actions], concession

    def apply_budget_reward(self, concession, done):
        """Spend this step's concessions and return the rule's rewards."""
        discount = self.gamma**self._t
        prev_budget = self.budget
        self.budget = prev_budget - discount * concession

        reward = self.rule.fn(
            BudgetStep(
                budget=self.budget.copy(),
                prev_budget=prev_budget,
                concession=concession,
                t=self._t,
                discount=discount,
                done=done,
            )
        )
        self._t += 1

        if self.common_reward:
            return float(reward)
        reward = np.asarray(reward, dtype=float)
        if reward.shape != (self.n_agents,):
            raise ValueError(
                f"budget rule '{self.rule_name}' returned shape {reward.shape}, "
                f"expected ({self.n_agents},)"
            )
        return reward.tolist()

    def step(self, action):
        """Takes (actions, concession); returns obss, reward, terminated, truncated, info"""
        actions, concession = self._split_action(action)
        _, env_rewards, done, truncated, info = self._env.step(actions)
        episode_end = done or truncated

        t = self._t
        reward = self.apply_budget_reward(concession, done=episode_end)
        # Built after the budget update, so the next action sees b_{t+1}.
        self._obs = self._observe()

        env_rewards = np.asarray(env_rewards, dtype=float)
        self._env_return += env_rewards
        self._env_disc_return += self.gamma**t * env_rewards
        if self._first_violation_t is None and (self.budget < 0).any():
            self._first_violation_t = t

        info = dict(info)
        # Per-step arrays for the runner to store in the batch (it pops them
        # before summing info into stats).
        info["budget_step"] = {
            "concession": tuple(float(d) for d in concession),
            "env_reward": tuple(float(r) for r in env_rewards),
        }
        if self._matrix is not None:
            info.update(self._matrix_step_stats(actions))
        if episode_end:
            info.update(self._episode_stats())
        return self._obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        _, info = self._env.reset(seed=seed, options=options)
        self._reset_budget()
        self._obs = self._observe()
        return self._obs, info

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self._obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        return self._obs[agent_id]

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return self._env.get_obs_size() + self._budget_obs_size()

    def _state_budget(self):
        # The state carries the full budget once whenever agents observe any of it.
        return self.budget if self.observe_budget != "none" else np.zeros(0)

    def get_state(self):
        return np.concatenate([self._env.get_state(), self._state_budget()]).astype(
            np.float32
        )

    def get_state_size(self):
        """Returns the shape of the state"""
        return self._env.get_state_size() + self._state_budget().size

    def get_avail_actions(self):
        return self._env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        return self._env.get_avail_agent_actions(agent_id)

    def get_total_actions(self):
        return self._env.get_total_actions()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        return self._env.seed(seed)

    def save_replay(self):
        self._env.save_replay()

    def get_stats(self):
        return self._env.get_stats()
