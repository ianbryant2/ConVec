"""One-step gymnasium environments wrapping games.Game, for epymarl.

Each generator in games.py is registered as a family of single-step
environments (the uoe-agents/matrix-games pattern): a Tuple action space
with one Discrete entry per agent, a constant dummy observation, and
per-agent rewards that `gymma` scalarises when common_reward=True.

Run from the epymarl directory with this project's root on PYTHONPATH:

    PYTHONPATH=/path/to/PAC-action_scaling python src/main.py \
        --config=qmix --env-config=gymma with \
        env_args.time_limit=1 env_args.key="matrix_envs:climbing-2p-v0" \
        reward_scalarisation="mean"

reward_scalarisation="mean" makes the logged return equal the game table's
value (the default "sum" multiplies it by num_agents). stag_hunt and the
decorrelated games are general-sum: run them with common_reward=False and
an algorithm that supports it.

Every step's info dict carries one-hot failure-mode flags from
Game.classify, which epymarl's runners aggregate and log as e.g.
`test_optimal_mean`.
"""

import gymnasium as gym
import numpy as np

from .games import (Game, anti_coordination, budget_table1, climbing, decorrelate,
                   kwise, needle, overestimation_trap, penalty_game,
                   risky_coordination, stag_hunt)

LABELS = ("optimal", "dominated_nash", "miscoordination", "other")


class OneStepGameEnv(gym.Env):
    """A games.Game as a single-step multi-agent gymnasium environment.

    The episode is one joint action: step returns the (possibly noisy)
    per-agent rewards from Game.sample and terminates. Observations are a
    constant zero per agent. Exposes n_agents, as epymarl's GymmaWrapper
    requires.
    """

    metadata = {"render_modes": []}

    def __init__(self, game_fn, deterministic=False, **game_kwargs):
        """
        Args:
            game_fn (Callable[..., Game]): generator called once with
                game_kwargs to build the (fixed) game. Generators with
                construction randomness must seed it internally so every
                instance plays the identical game (see the seeded factories
                below).
            deterministic (bool): return the exact Game.payoff instead of
                the noisy Game.sample; toggle from epymarl with
                env_args.deterministic=True.
        """
        self.game = game_fn(**game_kwargs)
        self.deterministic = deterministic
        self.n_agents = self.game.num_agents
        self.action_space = gym.spaces.Tuple(
            [gym.spaces.Discrete(d) for d in self.game.action_dims])
        obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Tuple(
            [obs_space for _ in range(self.n_agents)])

        # Failure-mode label of every joint action, computed once up front.
        joints = np.stack(np.indices(self.game.action_dims), axis=-1)
        self._labels = self.game.classify(joints)
        self._last_joint = None

    def _make_obs(self):
        return tuple(np.zeros(1, dtype=np.float32) for _ in range(self.n_agents))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._make_obs(), {}

    def step(self, actions):
        joint = np.asarray([int(a) for a in actions])
        self._last_joint = joint
        if self.deterministic:
            rewards = self.game.payoff(joint)
        else:
            # self.np_random carries the reset(seed=...) seed into the noise.
            rewards = self.game.sample(joint, rng=self.np_random)
        label = self._labels[tuple(joint)]
        info = {name: float(name == label) for name in LABELS}
        return self._make_obs(), [float(r) for r in rewards], True, False, info

    def render(self):
        if self._last_joint is not None:
            print(f"joint action {tuple(self._last_joint)}: "
                  f"{self._labels[tuple(self._last_joint)]}")

    def seed(self, seed=None):
        # GymmaWrapper prefers unwrapped.seed(seed); route it to gymnasium's
        # standard np_random seeding.
        gym.Env.reset(self, seed=seed)
        return [seed]


# ------------------------------------------------------------ seeded factories

def decorrelated_climbing(num_agents=2, rho=0.5, seed=0):
    """Failure (F) spectrum: climbing with off-optimum rewards decorrelated
    to rho. Fixed seed so every instance is the same game."""
    return decorrelate(climbing(num_agents), rho, rng=np.random.default_rng(seed))


def seeded_overestimation_trap(seed=0, **kwargs):
    """Failure (G) trap with the noisy-cell mask fixed by seed."""
    return overestimation_trap(rng=np.random.default_rng(seed), **kwargs)


def kwise_game(num_agents=4, num_actions=3, order=3, mix=1.0, num_groups=None,
               game_seed=0):
    """Failure (B) random instance: `mix` of the value variance sits in pure
    order-way interaction terms among `num_groups` random agent groups (see
    games.kwise). game_seed fixes the table and is deliberately separate
    from epymarl's training seed."""
    rng = np.random.default_rng(game_seed)
    table = kwise(order=order, mix=mix, num_groups=num_groups,
                  rng=rng)((num_actions,) * num_agents)
    return Game.common(table)


# ---------------------------------------------------------------- registration

def register_game(env_id, game_fn, **game_kwargs):
    """Register one game as a single-step env; use for parameter sweeps.

    Example:
        for p in (0, -4, -8, -12):
            register_game(f"risky-coordination-p{-p}-2p-v0",
                          risky_coordination, penalty=float(p))
    """
    gym.register(env_id, entry_point="matrix_envs:OneStepGameEnv",
                 kwargs={"game_fn": game_fn, **game_kwargs},
                 disable_env_checker=True)


_TABLE = [
    # Table 1 has conflicting individual rewards / Table 1 包含有冲突的个体奖励。
    ("budget-table1-3p-v0", budget_table1, {}),
    # (A) relative overgeneralization
    *[(f"risky-coordination-{n}p-v0", risky_coordination, {"num_agents": n})
      for n in (2, 3, 4)],
    # (A), and (B) representational ceiling at 3+ agents
    *[(f"climbing-{n}p-v0", climbing, {"num_agents": n}) for n in (2, 3, 4)],
    # (C) tied optima with cliffs; k sweep matches matrixgames:penalty-k-*
    *[(f"penalty-{k}-2p-v0", penalty_game, {"cliff": -float(k)})
      for k in (0, 25, 50, 75, 100)],
    ("penalty-100-3p-v0", penalty_game, {"num_agents": 3}),
    # (C) many tied optima
    *[(f"anti-coordination-{n}p-v0", anti_coordination, {"num_agents": n})
      for n in (2, 3, 4)],
    # (F) no-conflict but general-sum: needs common_reward=False
    *[(f"stag-hunt-{n}p-v0", stag_hunt, {"num_agents": n}) for n in (2, 3, 4)],
    # (F) spectrum, general-sum: needs common_reward=False
    *[(f"decorrelated-climbing-rho{int(rho * 100)}-2p-v0",
       decorrelated_climbing, {"rho": rho}) for rho in (0.0, 0.5, 0.9)],
    # (H) sparse optimum with an opt-out
    *[(f"needle-{n}p-v0", needle, {"num_agents": n}) for n in (2, 3, 4)],
    # (G) noisy mediocrity vs quiet optimum (stochastic rewards)
    *[(f"overestimation-trap-{n}p-v0", seeded_overestimation_trap,
       {"num_agents": n}) for n in (2, 3)],
    # (B) representational ceiling, random instances: value variance purely in
    # order-k interaction terms over t random agent groups (o = order,
    # t = num_groups, s = game seed; 3 actions per agent)
    *[(f"kwise-o{order}-t4-4p-s{s}-v0", kwise_game,
       {"num_agents": 4, "order": order, "num_groups": 4, "game_seed": s})
      for order in (2, 3) for s in (0, 1, 2)],
]

for _env_id, _game_fn, _kwargs in _TABLE:
    register_game(_env_id, _game_fn, **_kwargs)
