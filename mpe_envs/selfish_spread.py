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

layout="race" replaces simple_spread's uniform spawns with a race for one
landmark. On a board of [-board_scale, board_scale]^2 the landmarks sit in a row
on y = 0 in the right half, landmark_gap apart, starting just right of the
centre line, and the agents spawn uniformly in the left half at least
min_agent_gap apart (default: the collision distance). Every agent's nearest
landmark is then the first one in the row, and whoever gives way travels on
along the row to a further one. The board is not walled: board_scale only sets
where things spawn.

landmark_gap must be at least twice the collision distance, so the circles of
that radius around the landmarks don't overlap: agents sitting on neighbouring
landmarks don't collide, and an agent passing between two of them can do so
without colliding with either.
"""
import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.mpe.simple_spread.simple_spread import Scenario as SpreadScenario
from pettingzoo.utils.conversions import parallel_wrapper_fn


DEFAULT_COLLISION_DISTANCE = 0.3   # simple_spread's: two agent sizes of 0.15
LAYOUTS = (None, "race")


class Scenario(SpreadScenario):
    def __init__(self, collision_penalty=1.0, collision_distance=None, layout=None,
                 board_scale=1.0, landmark_gap=0.6, min_agent_gap=None):
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
        self.collision_penalty = collision_penalty
        self.collision_distance = collision_distance
        self.layout = layout
        self.board_scale = board_scale
        self.landmark_gap = landmark_gap
        d = DEFAULT_COLLISION_DISTANCE if collision_distance is None else collision_distance
        self.min_agent_gap = d if min_agent_gap is None else min_agent_gap
        if layout == "race" and landmark_gap < 2 * d:
            raise ValueError(f"landmark_gap {landmark_gap} must be at least twice the "
                             f"collision distance {d}")

    def reset_world(self, world, np_random):
        super().reset_world(world, np_random)
        if self.layout != "race":
            return
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
        layout=None,
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
            collision_distance=collision_distance,
            layout=layout,
            board_scale=board_scale,
            landmark_gap=landmark_gap,
            min_agent_gap=min_agent_gap,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario(collision_penalty, collision_distance, layout, board_scale,
                            landmark_gap, min_agent_gap)
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
