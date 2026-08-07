"""
Reward wrapper for LBF that replaces the level/food-proportional reward with
a shared Pareto-coordination signal:

- Agents that successfully load food together each get +1.
- Agents whose load attempt fails (combined level too low) each get -0.6.
- Agents that did not attempt to load food that step get 0.

The underlying ForagingEnv only makes a failed load distinguishable from
"did nothing" if it was created with a positive `penalty` (the default is
0.0, in which case both cases yield a reward of 0). This wrapper enforces
that requirement so remapping is unambiguous.

With `common_reward=True`, the whole step collapses to a single flat
scalar shared by every agent: +1 if any load succeeded this step, -0.6 if
none succeeded but at least one failed, else 0. It's returned as a plain
float rather than a per-agent list so that epymarl's GymmaWrapper (which
only aggregates when the reward is iterable, see envs/gymma.py) passes it
through unchanged regardless of `reward_scalarisation`.
"""
import gymnasium as gym


class ParetoForagingWrapper(gym.Wrapper):
    SUCCESS_REWARD = 1.0
    FAILURE_PENALTY = -0.6

    def __init__(self, env: gym.Env, common_reward: bool = False):
        penalty = getattr(env.unwrapped, "penalty", 0.0)
        if penalty <= 0:
            raise ValueError(
                "ParetoForagingWrapper requires the wrapped env to be created with a "
                "positive `penalty` so failed loads can be distinguished from agents "
                "that took no action this step. Pass penalty=... to gym.make()."
            )
        super().__init__(env)
        self.common_reward = common_reward

    def reset(self, seed, options):
        a, b = self.env.reset(seed, options)
        print(a, b)
        return a, b

    def step(self, actions):
        obs, rewards, done, truncated, info = self.env.step(actions)
        reward = 0
        for r in rewards:
            if r > 0:
                reward = 1
                break
            elif r < 0:
                reward = -.6
                break
        return obs, reward, done, truncated, info


