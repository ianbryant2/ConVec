"""Gymnasium registrations for PettingZoo MPE simple_spread at several team sizes.

PettingZoo's simple_spread_v3 takes N as a constructor kwarg, but epymarl's
auto-registered key (pz-mpe-simple-spread-v3) bakes nothing in, so every N
would share one sacred results dir. Registering one id per N here gives each
team size its own `mpe_envs:SimpleSpread-{N}ag-v0` key and results dir, the
same pattern as matrix_envs.

The entry point is epymarl's own PettingZooWrapper, resolved lazily at
gym.make time, so this module needs epymarl/src on sys.path only when an env
is actually constructed (true for training subprocesses launched by
scripts/sweeps), not when imported for id listing or results analysis.

Run from the epymarl directory with this project's root on PYTHONPATH:

    PYTHONPATH=/path/to/PAC-action_scaling python src/main.py \
        --config=qmix --env-config=gymma with \
        env_args.time_limit=25 env_args.key="mpe_envs:SimpleSpread-4ag-v0"

Episodes truncate at PettingZoo's internal max_cycles (default 25), so keep
env_args.time_limit in sync; to run longer episodes pass both, e.g.
env_args.time_limit=50 env_args.max_cycles=50 (extra env_args flow through
GymmaWrapper into the PettingZoo constructor, as do local_ratio etc.).
"""

import gymnasium as gym

# Collision distances with their own SelfishSpread ids, e.g.
# mpe_envs:SelfishSpread-3ag-col0.5-v0. The plain SelfishSpread-{N}ag-v0 keeps
# simple_spread's default (0.3, the sum of two agents' sizes); add a value here
# to get an id for it.
COLLISION_DISTANCES = (0.3, 0.4, 0.5, 0.6, 0.8, 1.0)

for n in range(2, 11):
    gym.register(
        f"SimpleSpread-{n}ag-v0",
        entry_point="envs.pz_wrapper:PettingZooWrapper",
        kwargs={"lib_name": "mpe", "env_name": "simple_spread_v3", "N": n},
    )
    # Individual rewards: own distance to nearest landmark + own collisions
    # (selfish_spread.py). Set the penalty with env_args.collision_penalty.
    gym.register(
        f"SelfishSpread-{n}ag-v0",
        entry_point="envs.pz_wrapper:PettingZooWrapper",
        kwargs={"module": "mpe_envs.selfish_spread", "N": n},
    )
    # The same, counting two agents as collided when closer than d. The
    # distance is in the id so each value gets its own sacred results dir.
    for d in COLLISION_DISTANCES:
        gym.register(
            f"SelfishSpread-{n}ag-col{d:g}-v0",
            entry_point="envs.pz_wrapper:PettingZooWrapper",
            kwargs={"module": "mpe_envs.selfish_spread", "N": n, "collision_distance": d},
        )
