"""Gymnasium registration for the SelfishSpread race (selfish_spread.py).
Episodes end at max_cycles, so run it with env_args.time_limit=50."""

import gymnasium as gym

gym.register(
    "SelfishSpread-3ag-race-v0",
    entry_point="envs.pz_wrapper:PettingZooWrapper",
    kwargs={"module": "mpe_envs.selfish_spread", "N": 3, "board_scale": 3.0,
            "landmark_gap": 1.2, "max_cycles": 50},
)
