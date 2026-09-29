"""lb-foraging's interactive human play, with a trained PAC policy's pi and Q shown.

Controls are unchanged from lb-foraging/human_play.py -- you drive every agent
yourself. The only addition is that each step prints, for every agent, the saved
policy's action distribution and the saved critic's Q-values, so you can see what
the trained models think of the state you are in.

You can control the interaction with the following keys:
- Arrow keys: move the current agent
- L: load food
- K: load food and keep the agent loading
- SPACE: do nothing
- TAB: change the current agent
- R: reset the environment
- H: show help
- D: display agent info (per step)
- ESC: exit

Example:
    python scripts/human_play.py \
        --checkpoint "epymarl/results/models/pac_sarsa_ns_seed0_lbforaging:Foraging-5x5-2p-1f-pen-v3_2026-08-07 16:39:50.296886"
"""

from argparse import ArgumentParser
from functools import partial
import json
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
import warnings

import numpy as np
import torch as th

PROJECT = Path(__file__).resolve().parent.parent
EPYMARL = PROJECT / "epymarl"
sys.path.insert(0, str(EPYMARL / "src"))

from components.episode_buffer import EpisodeBatch  # noqa: E402
from components.transforms import OneHot  # noqa: E402
from controllers import REGISTRY as mac_REGISTRY  # noqa: E402
from envs import REGISTRY as env_REGISTRY  # noqa: E402
from lbforaging.foraging.environment import Action  # noqa: E402
from modules.critics import REGISTRY as critic_REGISTRY  # noqa: E402
from modules.critics import register_pac_critics  # noqa: E402

ACTION_NAMES = [a.name for a in Action]

# Model dirs are named "{name}_seed{seed}_{env key}_{datetime}" (see run.py).
MODEL_DIR_RE = re.compile(r"^(?P<name>.+?)_seed(?P<seed>\d+)_(?P<key>.+?)_(?P<ts>\d{4}-\d\d-\d\d .*)$")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="a results/models/<unique_token> dir, or one numbered timestep dir inside it")
    parser.add_argument("--load-step", type=int, default=0,
                        help="timestep checkpoint to load (0 = latest); ignored if --checkpoint names one")
    parser.add_argument("--config-run", type=str, default=None,
                        help="sacred run dir with config.json (auto-detected from the checkpoint name)")
    parser.add_argument("--seed", type=int, default=None, help="env seed (default: the training seed)")
    parser.add_argument("--display_info", action="store_true", default=True,
                        help="display agent info per step")
    parser.add_argument("--dry-run", type=int, default=0, metavar="N",
                        help="open no window; play N steps with random actions (checks a checkpoint loads)")
    return parser.parse_args()


def find_config(checkpoint: Path, override: str | None) -> dict:
    """Locate the sacred config.json that produced a checkpoint dir."""
    if override:
        return json.loads((Path(override) / "config.json").read_text())

    match = MODEL_DIR_RE.match(checkpoint.name)
    if not match:
        raise SystemExit(f"Cannot parse '{checkpoint.name}'; pass --config-run explicitly.")

    key_dir = EPYMARL / "results" / "sacred" / match["name"] / match["key"]
    seed = int(match["seed"])
    for run_dir in sorted(key_dir.glob("*"), key=lambda p: p.name):
        if not run_dir.name.isdigit():
            continue
        config = json.loads((run_dir / "config.json").read_text())
        if config["seed"] == seed:
            print(f"Using config from {run_dir}")
            return config
    raise SystemExit(f"No sacred run under {key_dir} with seed {seed}; pass --config-run.")


def resolve_step(checkpoint: Path, load_step: int) -> Path:
    steps = [int(p.name) for p in checkpoint.iterdir() if p.is_dir() and p.name.isdigit()]
    if not steps:
        raise SystemExit(f"No numbered timestep dirs in {checkpoint}")
    step = max(steps) if load_step == 0 else min(steps, key=lambda x: abs(x - load_step))
    return checkpoint / str(step)


def other_action_index(joint_actions, agent, n_actions):
    """Row in the critic's enumerated other-agent joint actions.

    generate_other_actions() builds them with itertools.product over the agents
    j != i in ascending order, so the last agent varies fastest.
    """
    index = 0
    for j, action in enumerate(joint_actions):
        if j != agent:
            index = index * n_actions + int(action)
    return index


class InteractivePACEnv:
    """lb-foraging interactive play with a trained policy and critic read out."""

    def __init__(self, args):
        checkpoint = Path(args.checkpoint)
        if checkpoint.name.isdigit():
            # A single timestep dir was given; the config is named after its parent.
            checkpoint, model_path = checkpoint.parent, checkpoint
        else:
            model_path = resolve_step(checkpoint, args.load_step)
        config = find_config(checkpoint, args.config_run)

        self.cfg = SimpleNamespace(**config)
        self.cfg.use_cuda = False
        self.cfg.device = "cpu"

        self.dry_run = args.dry_run
        env_args = dict(config["env_args"])
        env_args.pop("seed", None)
        self.env = env_REGISTRY[config["env"]](
            **env_args,
            seed=args.seed if args.seed is not None else config["seed"],
            common_reward=config["common_reward"],
            reward_scalarisation=config["reward_scalarisation"],
            render_mode=None if self.dry_run else "human",
        )

        env_info = self.env.get_env_info()
        self.n_agents = env_info["n_agents"]
        self.n_actions = env_info["n_actions"]
        self.episode_limit = env_info["episode_limit"]
        self.cfg.n_agents = self.n_agents
        self.cfg.n_actions = self.n_actions
        self.cfg.state_shape = env_info["state_shape"]

        self._build_models(env_info, model_path)

        self.running = True
        self.current_agent_index = 0
        self.current_action = None

        self.loading_agents = []
        self.t = 0
        self.ep_returns = np.zeros(self.n_agents)
        self.reset = False

        self.display_info = args.display_info

        obss, _ = self._reset_env()

        if self.dry_run:
            self._display_info(obss, [0] * self.n_agents, False)
            self._random_cycle()
            return

        self.env.render()
        self.env._env.unwrapped.viewer.window.on_key_press = self._key_press

        if self.display_info:
            self._display_info(obss, [0] * self.n_agents, False)

        self._cycle()

    # -- model setup ------------------------------------------------------

    def _build_models(self, env_info, model_path):
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (self.n_actions,), "group": "agents", "dtype": th.int},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
            "reward": {"vshape": (1,) if self.cfg.common_reward else (self.n_agents,)},
        }
        groups = {"agents": self.n_agents}
        preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=self.n_actions)])}
        self.new_batch = partial(
            EpisodeBatch, scheme, groups, 1, self.episode_limit + 1,
            preprocess=preprocess, device="cpu",
        )
        # EpisodeBatch expands the scheme with actions_onehot; the mac and critic
        # are built against that expanded scheme, exactly as run.py does.
        full_scheme = self.new_batch().scheme

        self.mac = mac_REGISTRY[self.cfg.mac](full_scheme, groups, self.cfg)
        self.mac.load_models(str(model_path))

        register_pac_critics()
        self.critic = critic_REGISTRY[self.cfg.critic_type](full_scheme, self.cfg)
        self.critic.load_state_dict(th.load(f"{model_path}/critic.th", map_location="cpu"))
        print(f"Loaded policy and critic from {model_path}")

    def _reset_env(self):
        obss, info = self.env.reset()
        self.batch = self.new_batch()
        self.mac.init_hidden(batch_size=1)
        return obss, info

    def _evaluate(self):
        """pi and Q from the saved models for the state the env is in right now."""
        self.batch.update(
            {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            },
            ts=self.t,
        )
        with th.no_grad():
            pi = self.mac.forward(self.batch, t=self.t, test_mode=True)[0]
            # (n_agents, n_other_joint_actions, n_actions)
            q_all = self.critic(self.batch, t=self.t, compute_all=True)[0][0, 0]
        return pi, q_all

    # -- display ----------------------------------------------------------

    def _get_current_agent_info(self):
        agent_level = self.env._env.unwrapped.players[self.current_agent_index].level
        x, y = self.env._env.unwrapped.players[self.current_agent_index].position
        return f"Agent {self.current_agent_index + 1} (Level {agent_level}, at row {x + 1}, col {y + 1})"

    def _row(self, label, values, note=""):
        cells = "".join(f"{v:>9.3f}" for v in values)
        best = ACTION_NAMES[int(np.argmax(values))]
        return f"\t  {label:<18}{cells}   best={best:<6}{note}"

    def _display_models(self):
        """Print pi and Q for every agent in the current state."""
        pi, q_all = self._evaluate()

        # The actions the other agents will take if you press a key now: everyone
        # holds at NONE except agents you left loading with K.
        default_actions = [
            Action.LOAD.value if i in self.loading_agents else Action.NONE.value
            for i in range(self.n_agents)
        ]

        header = "".join(f"{name:>9}" for name in ACTION_NAMES[: self.n_actions])
        print(f"\t{'':<18}{header}")
        for i in range(self.n_agents):
            e = other_action_index(default_actions, i, self.n_actions)
            others = ", ".join(
                ACTION_NAMES[default_actions[j]] for j in range(self.n_agents) if j != i
            )
            print(f"\tAgent {i + 1}:")
            print(self._row("pi", pi[i].numpy()))
            print(self._row("Q | others hold", q_all[i, e].numpy(), f"others = {others}"))
            print(self._row("max_a- Q", q_all[i].max(dim=0)[0].numpy(), "best-case others"))

    def _display_info(self, obss, rews, done):
        print(f"Step {self.t}:")
        print(f"\tSelected: {self._get_current_agent_info()}")
        if self.loading_agents:
            print(f"\tLoading: {[i + 1 for i in self.loading_agents]}")
        print(f"\tObs: {obss[self.current_agent_index]}")
        print(f"\tAll Rews: {rews}")
        print(f"\tDone: {done}")
        if not done:
            self._display_models()
        print()

    def _help(self):
        print("Use the arrow keys to move the current agent")
        print("Use the L key to load food")
        print("Use the K key to load food and keep the agent loading")
        print("Use the SPACE key to do nothing")
        print("Press TAB to change the current agent")
        print("Press R to reset the environment")
        print("Press H to show help")
        print("Press D to display agent info")
        print("Press ESC to exit")
        print()

    # -- controls (unchanged from lb-foraging/human_play.py) ---------------

    def _increment_current_agent_index(self, index: int):
        index += 1
        if index == self.n_agents:
            index = 0
        return index

    def _key_press(self, k, mod):
        from pyglet.window import key

        if k == key.LEFT:
            self.current_action = Action.WEST
        elif k == key.RIGHT:
            self.current_action = Action.EAST
        elif k == key.DOWN:
            self.current_action = Action.SOUTH
        elif k == key.UP:
            self.current_action = Action.NORTH
        elif k == key.L:
            self.current_action = Action.LOAD
        elif k == key.K:
            self.current_action = Action.LOAD
            self.loading_agents.append(self.current_agent_index)
        elif k == key.SPACE:
            self.current_action = Action.NONE
        elif k == key.TAB:
            self.current_action = None
            self.current_agent_index = self._increment_current_agent_index(
                self.current_agent_index
            )
            if self.display_info:
                print(f"Now selected: {self._get_current_agent_info()}")
        elif k == key.R:
            self.current_action = None
            self.reset = True
        elif k == key.H:
            self.current_action = None
            self._help()
        elif k == key.D:
            self.current_action = None
            self.display_info = not self.display_info
        elif k == key.ESCAPE:
            self.running = False
        else:
            self.current_action = None
            warnings.warn(f"Key {k} not recognized")

        if k in [key.LEFT, key.RIGHT, key.DOWN, key.UP, key.L, key.SPACE]:
            if self.current_agent_index in self.loading_agents:
                self.loading_agents.remove(self.current_agent_index)

    def _take_step(self):
        actions = [
            Action.NONE if i not in self.loading_agents else Action.LOAD
            for i in range(self.n_agents)
        ]
        actions[self.current_agent_index] = self.current_action

        # Recorded so the models see the right history under configs that feed
        # the last action back in (obs_last_action).
        action_values = [act.value for act in actions]
        self.batch.update(
            {"actions": th.tensor(action_values).view(1, self.n_agents, 1)}, ts=self.t
        )

        obss, rews, done, trunc, info = self.env.step(action_values)
        self.ep_returns = self.ep_returns + np.asarray(rews)
        self.t += 1

        if self.display_info:
            self._display_info(obss, rews, done or trunc)

        if done or trunc or self.t >= self.episode_limit:
            self.reset = True

        self.current_action = None

    def _cycle(self):
        while self.running:
            if self.reset:
                if self.display_info:
                    print(f"Finished episode with episodic returns: {np.round(self.ep_returns, 3)}")
                    print()
                obss, _ = self._reset_env()
                self.reset = False
                self.ep_returns = np.zeros(self.n_agents)
                self.loading_agents = []
                self.t = 0

                if self.display_info:
                    self._display_info(obss, [0] * self.n_agents, False)

            if self.current_action is not None:
                self._take_step()
            self.env.render()
            time.sleep(0.02)
        self.env.close()

    def _random_cycle(self):
        rng = np.random.default_rng(0)
        for _ in range(self.dry_run):
            if self.reset:
                obss, _ = self._reset_env()
                self.reset = False
                self.ep_returns = np.zeros(self.n_agents)
                self.loading_agents = []
                self.t = 0
                self._display_info(obss, [0] * self.n_agents, False)
            self.current_action = Action(int(rng.integers(self.n_actions)))
            self._take_step()


if __name__ == "__main__":
    InteractivePACEnv(parse_args())
