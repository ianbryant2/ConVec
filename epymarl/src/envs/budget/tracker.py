"""Per-env concession budget bookkeeping for the budget runners."""
import gymnasium as gym
import numpy as np

from .rules import BudgetStep, get_rule


def concession_spread(charged):
    """How evenly each agent's concession is spread over an episode: the
    normalised entropy of its charges, 0 (one step) to 1 (every step equally).

    charged: (T, n_agents). Returns (spread, valid), each (n_agents,); valid
    is False when an agent conceded nothing or T < 2.
    """
    c = np.clip(np.asarray(charged, dtype=float), 0, None)
    T = c.shape[0]
    total = c.sum(axis=0)
    valid = (total > 0) & (T > 1)
    p = np.divide(c, total, out=np.zeros_like(c), where=total > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=0)
    spread = np.where(valid, entropy / np.log(max(T, 2)), 0.0)
    return np.clip(spread, 0.0, 1.0) + 0.0, valid  # + 0.0 turns -0.0 into 0.0


def budget_obs_size(observe_budget, n_agents):
    """Entries each agent's observation gains: nothing, its own b_i, or all of b."""
    return {"none": 0, "own": 1, "all": n_agents}[observe_budget]


def budget_state_size(observe_budget, n_agents):
    """The state carries the full budget once whenever agents observe any of it."""
    return 0 if observe_budget == "none" else n_agents


def matrix_game_reference(key, initial_budget, n_agents):
    """Feasible joint actions and best feasible welfare for a matrix game, else None."""
    if not (key or "").startswith("matrix_envs:"):
        return None
    env = gym.make(key)
    game = getattr(env.unwrapped, "game", None)
    env.close()
    if game is None:
        return None
    payoffs = game.rewards  # (|A_1|, ..., |A_n|, n_agents), expected payoffs
    utopian = payoffs.reshape(-1, n_agents).max(axis=0)
    feasible = (utopian - payoffs <= initial_budget).all(-1)
    welfare = payoffs.sum(-1)
    best = welfare[feasible].max() if feasible.any() else None
    return {"feasible": feasible, "welfare": welfare, "best": best}


class EpisodeBudget:
    """One env's budget: spends gamma^t * delta each step and returns the
    rule's reward."""

    def __init__(self, n_agents, common_reward, rule, initial_budget, gamma,
                 observe_budget, matrix=None):
        self.n_agents = n_agents

        self.rule_name = rule
        self.rule = get_rule(rule)
        if self.rule.common != common_reward:
            raise ValueError(
                f"budget rule '{rule}' produces "
                f"{'common' if self.rule.common else 'individual'} rewards but "
                f"common_reward={common_reward}; set common_reward={self.rule.common}"
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
        self._matrix = matrix
        self.reset()

    def reset(self):
        self.budget = self.initial_budget.copy()
        self.t = 0
        # Per-episode totals, reported in info when the episode ends.
        self._env_return = np.zeros(self.n_agents)
        self._env_disc_return = np.zeros(self.n_agents)
        self._first_violation_t = None
        self._charged = []  # per step: (n_agents,) budget charged, gamma^t * delta
        # Per step, for the concession model's training data.
        self.env_rewards = []
        self.concessions = []

    def observe(self, obs):
        """The env's observations with the budget each agent can see."""
        if self.observe_budget == "none":
            return obs
        if self.observe_budget == "own":
            return [np.append(o, self.budget[i]) for i, o in enumerate(obs)]
        return [np.concatenate([o, self.budget]) for o in obs]

    def state(self, state):
        """The env's state with the budget appended (unless it is unobserved)."""
        extra = self.budget if self.observe_budget != "none" else np.zeros(0)
        return np.concatenate([state, extra]).astype(np.float32)

    def step(self, actions, concession, env_rewards, done):
        """Spend this step's concessions; returns (reward, info)."""
        concession = np.asarray(concession, dtype=float)
        if concession.shape != (self.n_agents,):
            raise ValueError(
                f"concession has shape {concession.shape}, expected ({self.n_agents},)"
            )
        # The true delta_i = V*_i - Q*_i is >= 0 (Lemma 14 of the writeup), but a
        # learned estimate may dip below 0, as may the "td" delta's sample of it
        # in a stochastic env; it is spent unclamped so that its noise cancels
        # over an episode instead of adding up.
        if not np.all(np.isfinite(concession)):
            raise ValueError(f"concession must be finite, got {concession}")

        t = self.t
        reward = self._spend(concession, done)

        env_rewards = np.asarray(env_rewards, dtype=float)
        if env_rewards.shape != (self.n_agents,):
            raise ValueError(
                f"the env must report one reward per agent, got shape {env_rewards.shape}"
            )
        self._env_return += env_rewards
        self._env_disc_return += self.gamma**t * env_rewards
        if self._first_violation_t is None and (self.budget < 0).any():
            self._first_violation_t = t
        self.env_rewards.append(tuple(float(r) for r in env_rewards))
        self.concessions.append(tuple(float(d) for d in concession))

        info = {}
        if self._matrix is not None:
            info.update(self._matrix_step_stats(actions))
        if done:
            info.update(self._episode_stats())
        return reward, info

    def _spend(self, concession, done):
        """Spend this step's concessions and return the rule's rewards."""
        discount = self.gamma**self.t
        prev_budget = self.budget
        self.budget = prev_budget - discount * concession
        self._charged.append(discount * concession)

        reward = self.rule.fn(
            BudgetStep(
                budget=self.budget.copy(),
                prev_budget=prev_budget,
                concession=concession,
                t=self.t,
                discount=discount,
                done=done,
            )
        )
        self.t += 1

        if self.common_reward:
            return float(reward)
        reward = np.asarray(reward, dtype=float)
        if reward.shape != (self.n_agents,):
            raise ValueError(
                f"budget rule '{self.rule_name}' returned shape {reward.shape}, "
                f"expected ({self.n_agents},)"
            )
        return reward.tolist()

    def _matrix_step_stats(self, actions):
        """Scores a matrix-game joint action against the constrained optimum."""
        joint = tuple(int(a) for a in actions)
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
        """End-of-episode stats, logged as <key>_mean over episodes."""
        stats = {}
        spread, spread_valid = concession_spread(np.array(self._charged).reshape(-1, self.n_agents))
        for i in range(self.n_agents):
            # The runner logs sums over episodes / episode count, so the mean
            # spread over episodes where it is defined is
            # concession_spread_i_mean / concession_spread_i_valid_mean.
            stats[f"concession_spread_{i}"] = spread[i]
            stats[f"concession_spread_{i}_valid"] = spread_valid[i]
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
            self.t if self._first_violation_t is None else self._first_violation_t
        )
        return {k: float(v) for k, v in stats.items()}
