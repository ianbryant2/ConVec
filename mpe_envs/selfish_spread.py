"""Selfish spread race: simple_spread_v3 with individual rewards and a race
for one landmark.

    r_i = -min_l ||p_i - p_l|| - collision_penalty * (# agents colliding with i)

Landmarks sit in a row on y = 0 in the right half of the board, landmark_gap
apart; agents spawn in the left half. Every agent's nearest landmark is the
first one, so whoever gives way travels on to a further one.
"""
import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.mpe.simple_spread.simple_spread import Scenario as SpreadScenario
from pettingzoo.utils.conversions import parallel_wrapper_fn


COLLISION_DISTANCE = 0.3   # simple_spread's: two agent sizes of 0.15


class Scenario(SpreadScenario):
    def __init__(self, collision_penalty=1.0, board_scale=1.0, landmark_gap=0.6,
                 min_agent_gap=None):
        self.collision_penalty = collision_penalty
        self.board_scale = board_scale
        self.landmark_gap = landmark_gap
        self.min_agent_gap = COLLISION_DISTANCE if min_agent_gap is None else min_agent_gap
        if landmark_gap < 2 * COLLISION_DISTANCE:
            raise ValueError(f"landmark_gap {landmark_gap} must be at least twice the "
                             f"collision distance {COLLISION_DISTANCE}")

    def reset_world(self, world, np_random):
        # simple_spread's reset first (colours, velocities, and its random
        # draws), then the race positions over it.
        super().reset_world(world, np_random)
        s = self.board_scale
        x0 = 0.05 * s   # the row starts, and the agents' half ends, this far from the centre line
        for k, lm in enumerate(world.landmarks):
            lm.state.p_pos = np.array([x0 + k * self.landmark_gap, 0.0])
        placed = []
        for agent in world.agents:
            for _ in range(1000):
                p = np.array([np_random.uniform(-s, -x0), np_random.uniform(-s, s)])
                if all(np.linalg.norm(p - q) >= self.min_agent_gap for q in placed):
                    break
            else:
                raise RuntimeError("couldn't place the agents min_agent_gap apart")
            placed.append(p)
            agent.state.p_pos = p

    def reward(self, agent, world):
        nearest = min(
            np.linalg.norm(agent.state.p_pos - lm.state.p_pos)
            for lm in world.landmarks
        )
        collisions = sum(
            a is not agent and self.is_collision(a, agent) for a in world.agents
        )
        return -nearest - self.collision_penalty * collisions


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=3,
        collision_penalty=1.0,
        board_scale=1.0,
        landmark_gap=0.6,
        min_agent_gap=None,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
    ):
        EzPickle.__init__(
            self,
            N=N,
            collision_penalty=collision_penalty,
            board_scale=board_scale,
            landmark_gap=landmark_gap,
            min_agent_gap=min_agent_gap,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario(collision_penalty, board_scale, landmark_gap, min_agent_gap)
        world = scenario.make_world(N)
        # local_ratio=None: SimpleEnv then returns scenario.reward unmixed, with
        # no shared global term.
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=None,
            dynamic_rescaling=dynamic_rescaling,
        )
        self.metadata["name"] = "selfish_spread_v0"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
