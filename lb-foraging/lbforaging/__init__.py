from itertools import product

from gymnasium import register

from .pareto_foraging import ParetoForagingWrapper
from .rel_obs_foraging import RelativeObservationWrapper
from .foraging import ForagingEnv


def _rel_obs_entry_point(**kwargs):
    """`-rel_obs` ids: positions reported as signed offsets from each agent."""
    return RelativeObservationWrapper(ForagingEnv(**kwargs))


sizes = range(5, 6)
players = range(2, 4)
foods = range(1, 10)
max_food_level = [None]  # [None, 1]
coop = [True, False]
partial_obs = [True, False]
# Failed-load penalty, carried in the id so every run records it: "-pen0.3"
# is penalty 0.3, "-pen0.0" no penalty. The suffix-less and bare "-pen" ids are
# legacy, kept so older checkpoints still load: no suffix is 0.0, "-pen" is 0.6
# now, but some past "-pen" runs were trained at 0.1.
pens = [("", 0.0), ("-pen", 0.6)] + [(f"-pen{v}", v) for v in (0.0, 0.1, 0.3, 0.6)]
rel_obs = [True, False]  # [True, False]


for s, p, f, mfl, c, po, (pen_suffix, penalty), rel in product(
    sizes, players, foods, max_food_level, coop, partial_obs, pens, rel_obs
):
    register(
        id="Foraging{4}-{0}x{0}-{1}p-{2}f{3}{5}{6}{7}-v3".format(
            s,
            p,
            f,
            "-coop" if c else "",
            "-2s" if po else "",
            "-ind" if mfl else "",
            pen_suffix,
            "-rel_obs" if rel else "",
        ),
        entry_point=_rel_obs_entry_point if rel else "lbforaging.foraging:ForagingEnv",
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
            "penalty": penalty,
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

# for s in sizes:
#     for p in players:
#         for f in foods:
#             for c in coops:
#                 register(
#                     id="Foraging-{0}x{0}-{1}p-{2}f{3}-v3".format(s, p, f, "-coop" if c else ""),
#                     entry_point="lbforaging.foraging:ForagingEnv",
#                     kwargs={
#                         "players": p,
#                         "max_player_level": 3,
#                         "field_size": (s, s),
#                         "max_num_food": f,
#                         "sight": s,
#                         "max_episode_steps": 25,
#                         "force_coop": c,
#                         "max_food_level": None,
#                         "min_player_level" : 1,
#                         "min_food_level" : 1
#                     },
#                 )
#                 register(
#                     id="Foraging-{0}x{0}-{1}p-{2}f{3}-pareto-v3".format(s, p, f, "-coop" if c else ""),
#                     entry_point=_pareto_entry_point,
#                     kwargs={
#                         "players": p,
#                         "max_player_level": 3,
#                         "field_size": (s, s),
#                         "max_num_food": f,
#                         "sight": s,
#                         "max_episode_steps": 25,
#                         "force_coop": c,
#                         "max_food_level": None,
#                         "min_player_level" : 1,
#                         "min_food_level" : 1
#                     }
#                 )


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
