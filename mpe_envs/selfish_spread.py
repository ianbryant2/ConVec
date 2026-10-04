"""Selfish spread: simple_spread with individual rewards instead of a team reward.

Each agent is paid for its own distance to the nearest landmark and penalised
for its own collisions:

    r_i = -min_l ||p_i - p_l|| - collision_penalty * (# agents colliding with i)

In simple_spread the distance term is shared (per landmark, the distance to the
closest agent), so nobody gains from crowding a landmark someone already
covers. Here each agent wants the landmark nearest to itself, whoever else is
there, so agents that share a nearest landmark compete for it and one of them
has to settle for a worse one: a general-sum game rather than a coordination
game. Observations, dynamics and the world are those of simple_spread_v3.

collision_distance sets how close two agents must be to count as colliding in
the reward. The default (None) is simple_spread's: the sum of the two agents'
sizes, 0.15 + 0.15 = 0.3. It changes only the reward; the physics still keeps
agents apart using their sizes, so movement and rendering are unchanged.
"""
import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.mpe.simple_spread.simple_spread import Scenario as SpreadScenario
from pettingzoo.utils.conversions import parallel_wrapper_fn


class Scenario(SpreadScenario):
    def __init__(self, collision_penalty=1.0, collision_distance=None):
        self.collision_penalty = collision_penalty
        self.collision_distance = collision_distance

    def is_collision(self, agent1, agent2):
        if self.collision_distance is None:
            return super().is_collision(agent1, agent2)
        dist = np.linalg.norm(agent1.state.p_pos - agent2.state.p_pos)
        return dist < self.collision_distance

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
        collision_distance=None,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
    ):
        EzPickle.__init__(
            self,
            N=N,
            collision_penalty=collision_penalty,
            collision_distance=collision_distance,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario(collision_penalty, collision_distance)
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
