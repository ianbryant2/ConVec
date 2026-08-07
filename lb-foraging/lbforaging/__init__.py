from itertools import product

from gymnasium import register

from .pareto_foraging import ParetoForagingWrapper
from .foraging import ForagingEnv


sizes = range(5, 20)
players = range(2, 10)
foods = range(1, 10)
max_food_level = [None]  # [None, 1]
coop = [True, False]
partial_obs = [True, False]
pens = [True, False]  # [True, False]


for s, p, f, mfl, c, po, pen in product(
    sizes, players, foods, max_food_level, coop, partial_obs, pens
):
    register(
        id="Foraging{4}-{0}x{0}-{1}p-{2}f{3}{5}{6}-v3".format(
            s,
            p,
            f,
            "-coop" if c else "",
            "-2s" if po else "",
            "-ind" if mfl else "",
            "-pen" if pen else "",
        ),
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": p,
            "min_player_level": 1,
            "max_player_level": 2,
            "field_size": (s, s),
            "min_food_level": 1,
            "max_food_level": mfl,
            "max_num_food": f,
            "sight": 2 if po else s,
            "max_episode_steps": 50,
            "force_coop": c,
            "grid_observation": False,
            "penalty": 0.6 if pen else 0.0,
        },
    )

#Pareto Specific
sizes = [5]
players = [2, 3]
foods = [1]
coops = [True, False]

def _pareto_entry_point(**kwargs):
    kwargs.setdefault("penalty", 1.0)
    return ParetoForagingWrapper(ForagingEnv(**kwargs))

for s in sizes:
    for p in players:
        for f in foods:
            for c in coops:
                register(
                    id="Foraging-{0}x{0}-{1}p-{2}f{3}-v3".format(s, p, f, "-coop" if c else ""),
                    entry_point="lbforaging.foraging:ForagingEnv",
                    kwargs={
                        "players": p,
                        "max_player_level": 3,
                        "field_size": (s, s),
                        "max_num_food": f,
                        "sight": s,
                        "max_episode_steps": 25,
                        "force_coop": c,
                        "max_food_level": None,
                        "min_player_level" : 1,
                        "min_food_level" : 1
                    },
                )
                register(
                    id="Foraging-{0}x{0}-{1}p-{2}f{3}-pareto-v3".format(s, p, f, "-coop" if c else ""),
                    entry_point=_pareto_entry_point,
                    kwargs={
                        "players": p,
                        "max_player_level": 3,
                        "field_size": (s, s),
                        "max_num_food": f,
                        "sight": s,
                        "max_episode_steps": 25,
                        "force_coop": c,
                        "max_food_level": None,
                        "min_player_level" : 1,
                        "min_food_level" : 1
                    }
                )


def register_grid_envs():
    for s, p, f, mfl, c in product(sizes, players, foods, max_food_level, coop):
        for sight in range(1, s + 1):
            register(
                id="Foraging-grid{4}-{0}x{0}-{1}p-{2}f{3}{5}-v3".format(
                    s,
                    p,
                    f,
                    "-coop" if c else "",
                    "" if sight == s else f"-{sight}s",
                    "-ind" if mfl else "",
                ),
                entry_point="lbforaging.foraging:ForagingEnv",
                kwargs={
                    "players": p,
                    "min_player_level": 1,
                    "max_player_level": 2,
                    "field_size": (s, s),
                    "min_food_level": 1,
                    "max_food_level": mfl,
                    "max_num_food": f,
                    "sight": sight,
                    "max_episode_steps": 50,
                    "force_coop": c,
                    "grid_observation": True,
                },
            )
