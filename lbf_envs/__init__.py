"""Level-Based Foraging envs for this project, used as gymma key `lbf_envs:<id>`.

Foraging-5x5-3p-2f-budget-v0: 3 level-1 agents, 2 apples worth 1/2 each,
split among the agents loading it; fully observable, 25 steps. With budget
[0.55, 0.8, 0.8] the intended outcome is agent 0 eating one apple alone and
agents 1 and 2 sharing the other.
"""

import gymnasium as gym

gym.register(
    "Foraging-5x5-3p-2f-budget-v0",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "min_player_level": 1,
        "max_player_level": 1,
        "field_size": (5, 5),
        "min_food_level": 1,
        "max_food_level": 1,
        "max_num_food": 2,
        "sight": 5,
        "max_episode_steps": 25,
        "force_coop": False,
        "normalize_reward": True,
        "grid_observation": False,
        "penalty": 0.0,
    },
)
