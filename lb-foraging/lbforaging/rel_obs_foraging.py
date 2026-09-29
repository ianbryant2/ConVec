"""
Observation wrapper for LBF that rewrites every absolute (y, x) position in
the observation as a signed offset from the observing agent.

The base env reports positions in a neighbourhood frame centred on the
observing player (see `ForagingEnv._transform_to_neighborhood`), which for
full-sight envs coincides with absolute grid coordinates. This wrapper
replaces each entity's coordinate pair with

    (entity_y - self_y, entity_x - self_x)

so an agent reads "the food is 2 rows up and 1 column right" instead of
"the food is at (1, 3) and I am at (3, 2)". The offsets keep their sign, so
which side of the food the agent is on is recoverable; a value of 0 on an
axis means the agent is already aligned on it, and Manhattan distance to
the food is just `abs(dy) + abs(dx)`.

Food and player levels are passed through untouched, and the observing
agent stays first in the player block. Its own coordinate pair would be
(0, 0) by construction once everything is measured against it, so the two
entries are dropped and only its level is kept; every other slot keeps its
(y, x, level) shape. The resulting observation is exactly "own level, plus
the levels and relative distances of the other agents and the food", which
for the 2-player 1-food ids is 7 values rather than the base layout's 9.

Absent or out-of-sight entities are marked `-1, -1` by the base env, which
would be indistinguishable from a real offset of one row/column back.
They are re-marked here with `ABSENT`, one step outside the reachable
offset range, so the sentinel stays unambiguous under partial observability
(the `-2s` env ids) and whether or not agent levels are observed.
"""

import gymnasium as gym
import numpy as np

# The base env's "not visible / no such entity" marker, in its own frame.
_BASE_ABSENT = -1.0


class RelativeObservationWrapper(gym.ObservationWrapper):
    """Rewrites absolute positions in the observation as signed offsets."""

    def __init__(self, env: gym.Env):
        inner = env.unwrapped
        if inner._grid_observation:
            raise ValueError(
                "RelativeObservationWrapper only supports the vector observation "
                "layout, but the wrapped env was created with "
                "grid_observation=True, whose observation is a stack of spatial "
                "layers with no absolute positions to rewrite."
            )
        super().__init__(env)

        self.max_num_food = inner.max_num_food
        self.n_players = len(inner.players)
        self.player_obs_len = 3 if inner._observe_agent_levels else 2

        # Index of the observing agent's own (y, x); it always leads the
        # player block, so every other slot is measured against it.
        self.self_offset = 3 * self.max_num_food

        # Slot starts, each holding a (y, x) pair to rewrite.
        self.coord_starts = [3 * i for i in range(self.max_num_food)] + [
            self.self_offset + self.player_obs_len * i for i in range(self.n_players)
        ]

        # Largest offset reachable on a board this size, plus the sentinel
        # that sits just outside it.
        self.span = max(inner.rows, inner.cols) - 1
        self.absent = -(self.span + 1)

        # The observing agent's own offset is always (0, 0), so those two
        # entries are dropped; its level (when observed) is kept.
        base_len = 3 * self.max_num_food + self.player_obs_len * self.n_players
        self.keep = np.array(
            [
                i
                for i in range(base_len)
                if i not in (self.self_offset, self.self_offset + 1)
            ]
        )

        self.observation_space = gym.spaces.Tuple(
            tuple(self._relative_space(s) for s in env.observation_space)
        )

    def _relative_space(self, space):
        """`space` with coordinate bounds widened to signed offsets.

        Level bounds are copied straight from the base space so the food and
        player level ranges stay correct without re-deriving them here, and
        the observing agent's own coordinate entries are dropped alongside
        the observation itself.
        """
        low = np.array(space.low, dtype=np.float32)
        high = np.array(space.high, dtype=np.float32)
        for start in self.coord_starts:
            low[start : start + 2] = self.absent
            high[start : start + 2] = self.span
        low, high = low[self.keep], high[self.keep]
        return gym.spaces.Box(low=low, high=high, shape=low.shape, dtype=np.float32)

    def observation(self, observation):
        return tuple(self._to_relative(obs) for obs in observation)

    def _to_relative(self, obs):
        obs = np.array(obs, dtype=np.float32)
        self_y, self_x = obs[self.self_offset], obs[self.self_offset + 1]

        for start in self.coord_starts:
            y, x = obs[start], obs[start + 1]
            if y == _BASE_ABSENT and x == _BASE_ABSENT:
                obs[start] = obs[start + 1] = self.absent
            else:
                obs[start] = y - self_y
                obs[start + 1] = x - self_x

        return obs[self.keep]
