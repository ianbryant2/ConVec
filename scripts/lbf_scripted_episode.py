"""Per-step concession a saved critic charges in one scripted LBF episode.

Env lbf_envs:Foraging-5x5-3p-2f-budget-v0 (3 level-1 agents, 2 apples paying
1/2 each, split among the agents loading it). Scripted agents play the
intended split on one fixed two-apple layout (seed 10000): agent 0 eats one
apple alone, agents 1 and 2 load the other together. Every step is charged
with each critic as the budget runner would (budget_concession.py):

  td         V_i(s_t) - r_i,t - gamma (1 - done) V_i(s_t+1)
  advantage  V_i(s_t) - Q_i(s_t, a_t)

A critic is the concession_critic.pt the budget runner saves in its sacred dir
(give the file or the run dir). With several critics (e.g. one per seed) the
CSV has a row per critic, and the figure plots their mean shaded by one
standard error.

    python scripts/lbf_scripted_episode.py <run dir> [<run dir> ...] --csv out.csv --pdf out.pdf

Taken from main's scripts/lbf_feasibility.py (scripted agents, step kinds).
"""
import argparse
import csv
import itertools
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PROJECT / "epymarl" / "src"), str(PROJECT)]

from envs.gymma import GymmaWrapper  # noqa: E402

KEY = "lbf_envs:Foraging-5x5-3p-2f-budget-v0"
TIME_LIMIT = 25
N_AGENTS = 3
NONE, NORTH, SOUTH, WEST, EAST, LOAD = range(6)
MOVES = {NORTH: (-1, 0), SOUTH: (1, 0), WEST: (0, -1), EAST: (0, 1)}
# The example episode: layout seed and who loads which apple (apple index in
# np.argwhere order: {apple: [agents]}).
LAYOUT_SEED = 10_000
ASSIGNMENT = {0: [0], 1: [1, 2]}


# --- grid helpers ----------------------------------------------------------

def make_env(seed=0):
    return GymmaWrapper(KEY, TIME_LIMIT, None, seed, False, "sum")


def game(env):
    return env._env.unwrapped


def neighbours(cell, rows, cols):
    r, c = cell
    for dr, dc in MOVES.values():
        if 0 <= r + dr < rows and 0 <= c + dc < cols:
            yield r + dr, c + dc


def bfs(start, goals, walls, rows, cols):
    """(distance, first action) from start to the nearest goal, or (None, None)."""
    if start in goals:
        return 0, NONE
    seen = {start: None}
    queue = deque([(start, 0)])
    while queue:
        cell, d = queue.popleft()
        for action, (dr, dc) in MOVES.items():
            nxt = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols) or nxt in walls or nxt in seen:
                continue
            seen[nxt] = action if cell == start else seen[cell]
            if nxt in goals:
                return d + 1, seen[nxt]
            queue.append((nxt, d + 1))
    return None, None


def adjacent_cells(food, foods, rows, cols):
    return [n for n in neighbours(food, rows, cols) if n not in foods]


# --- scripted agents ---------------------------------------------------------

class Scripted:
    """Each group walks to distinct cells next to its apple and loads it
    together once every member is in place; agents in no group stay put."""

    def __init__(self, env, assignment):
        g = game(env)
        self.rows, self.cols = g.field.shape
        foods = [tuple(f) for f in np.argwhere(g.field > 0)]
        self.groups = {foods[k]: agents for k, agents in assignment.items()}

    def _cells(self, food, foods, n):
        cells = adjacent_cells(food, foods, self.rows, self.cols)
        # Cells next to another apple could load the wrong one; avoid them.
        clean = [c for c in cells
                 if not any(f != food and abs(c[0] - f[0]) + abs(c[1] - f[1]) == 1 for f in foods)]
        return clean if len(clean) >= n else cells

    def _step_to(self, pos, goal, foods, others):
        _, action = bfs(pos, {goal}, foods | others, self.rows, self.cols)
        if action is None:  # boxed in by agents: ignore them and try anyway
            _, action = bfs(pos, {goal}, foods, self.rows, self.cols)
        return NONE if action is None else action

    def _place(self, agents, cells, pos, foods):
        """Distinct cells for the group, minimising their total distance."""
        best = None
        for combo in itertools.permutations(cells, len(agents)):
            dists = [bfs(pos[a], {c}, foods, self.rows, self.cols)[0] for a, c in zip(agents, combo)]
            if None in dists:
                continue
            if best is None or sum(dists) < best[0]:
                best = (sum(dists), combo)
        return None if best is None else dict(zip(agents, best[1]))

    def act(self, env):
        g = game(env)
        foods = {tuple(f) for f in np.argwhere(g.field > 0)}
        pos = [tuple(p.position) for p in g.players]
        actions = [NONE] * N_AGENTS
        for food, agents in self.groups.items():
            if food not in foods:
                continue
            place = self._place(agents, self._cells(food, foods, len(agents)), pos, foods)
            if place is None:
                continue
            if all(pos[a] == cell for a, cell in place.items()):
                for a in agents:
                    actions[a] = LOAD
                continue
            for a, cell in place.items():
                others = {pos[j] for j in range(N_AGENTS) if j != a}
                actions[a] = NONE if pos[a] == cell else self._step_to(pos[a], cell, foods, others)
        return self._yield(actions, pos)

    @staticmethod
    def _yield(actions, pos):
        """LBF cancels every move into a cell another agent also ends up in,
        which deterministic agents would repeat forever; the later agent waits."""
        def dest(i):
            if actions[i] in MOVES:
                dr, dc = MOVES[actions[i]]
                return pos[i][0] + dr, pos[i][1] + dc
            return pos[i]

        changed = True
        while changed:
            changed = False
            for i in range(N_AGENTS):
                if actions[i] in MOVES and any(dest(j) == dest(i) for j in range(N_AGENTS)
                                               if j != i and (j < i or actions[j] not in MOVES)):
                    actions[i] = NONE
                    changed = True
        return actions


def play_scripted(seed=LAYOUT_SEED, assignment=ASSIGNMENT):
    """One scripted episode: per step, the critic's inputs (state and avail
    actions before and after, actions, rewards, done) and what step_kind needs
    (positions and apples before and after)."""
    env = make_env(seed)
    env.reset(seed=seed)
    g = game(env)
    policy = Scripted(env, assignment)
    traj = {"state": [], "avail": [], "actions": [], "reward": [], "pos": [], "foods": []}
    for t in range(TIME_LIMIT):
        actions = policy.act(env)
        traj["state"].append(env.get_state())
        traj["avail"].append(np.asarray(env.get_avail_actions()))
        traj["actions"].append(actions)
        traj["pos"].append([tuple(p.position) for p in g.players])
        traj["foods"].append([tuple(f) for f in np.argwhere(g.field > 0)])
        _, reward, done, truncated, _ = env.step(actions)
        traj["reward"].append([float(r) for r in reward])
        if done or truncated:
            break
    traj["pos"].append([tuple(p.position) for p in g.players])
    traj["foods"].append([tuple(f) for f in np.argwhere(g.field > 0)])
    traj["final_state"] = env.get_state()
    traj["final_avail"] = np.asarray(env.get_avail_actions())
    env.close()
    return traj


def step_kind(traj, k, i):
    """What step k of a scripted trajectory is for agent i. Loads are split by
    whether they eat the episode's last apple: that step is terminal, so its
    TD target is just the reward."""
    eaten = len(traj["foods"][k]) > len(traj["foods"][k + 1])
    if traj["reward"][k][i] > 0 or eaten:
        who = "own load" if traj["reward"][k][i] > 0 else "other's load"
        return f"{who}, {'last' if not traj['foods'][k + 1] else 'first'} apple"
    return "move" if traj["pos"][k + 1][i] != traj["pos"][k][i] else "stay"


# --- charges -------------------------------------------------------------------

def load_critic(path):
    """A saved concession critic: concession_critic.pt, or the run dir holding it."""
    import torch as th
    path = Path(path)
    if path.is_dir():
        path = path / "concession_critic.pt"
    return th.load(path, weights_only=False)


def charges(model, traj, gamma, delta="td"):
    """(n_steps, n_agents): the critic's charge at every step of traj, as the
    budget runner makes it (the episode ends at its last step)."""
    import torch as th
    n = len(traj["state"])
    states = traj["state"][1:] + [traj["final_state"]]
    avails = traj["avail"][1:] + [traj["final_avail"]]
    out = []
    with th.no_grad():
        for k in range(n):
            s = th.as_tensor(traj["state"][k][None], dtype=th.float32)
            avail = th.as_tensor(traj["avail"][k][None])
            if delta == "advantage":
                d = model(s, th.as_tensor([traj["actions"][k]]), avail, k)
            else:
                alive = 0.0 if k == n - 1 else 1.0
                nxt = th.as_tensor(states[k][None], dtype=th.float32)
                d = (model.values(s, avail, k) - th.as_tensor(traj["reward"][k], dtype=th.float32)
                     - gamma * alive * model.values(nxt, th.as_tensor(avails[k][None]), k + 1))
            out.append(np.asarray(d[0], dtype=float))
    return np.array(out)


def run_gamma(path):
    """gamma from the sacred run holding the critic, or None."""
    path = Path(path)
    config = (path if path.is_dir() else path.parent) / "config.json"
    return json.loads(config.read_text())["gamma"] if config.exists() else None


# --- figure ----------------------------------------------------------------------

def plot(traj, mean, sem, path):
    """One panel per agent: the mean charge at every step, shaded by one
    standard error over critics. Vertical lines mark loads (solid: the agent's
    own; dotted: another agent's)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series, ink, muted, grid = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42,  # embed TrueType so the text stays selectable
    })
    steps = np.arange(len(mean))
    fig, axes = plt.subplots(N_AGENTS, 1, figsize=(3.4, 4.5), sharex=True, sharey=True)
    for i, ax in enumerate(axes):
        for k in steps:
            kind = step_kind(traj, k, i)
            if "load" in kind:
                ax.axvline(k, color=muted, lw=0.7, ls="-" if kind.startswith("own") else ":", zorder=1)
        ax.axhline(0, color=muted, lw=0.5, zorder=1)
        ax.fill_between(steps, mean[:, i] - sem[:, i], mean[:, i] + sem[:, i],
                        color=series, alpha=0.2, lw=0, zorder=2)
        ax.plot(steps, mean[:, i], "o-", color=series, lw=1.2, ms=3, zorder=3)
        ax.set_title(f"Agent {i}", loc="left", color=ink, pad=3)
        ax.grid(axis="y", color=grid, lw=0.5, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(muted)
            ax.spines[s].set_linewidth(0.6)
        ax.tick_params(colors=muted, width=0.6, length=2.5)
        ax.set_xticks(steps)
        ax.set_ylabel("Concession", color=muted)
    axes[-1].set_xlabel("Episode step", color=muted)
    fig.tight_layout(pad=0.3, h_pad=0.8)
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("critics", nargs="+", help="concession_critic.pt files or the sacred run dirs holding them")
    parser.add_argument("--delta", choices=("td", "advantage"), default="td", help="the charge (default td)")
    parser.add_argument("--gamma", type=float, help="default: the first critic's run config, else 0.99")
    parser.add_argument("--csv", help="write one row per (critic, step, agent)")
    parser.add_argument("--pdf", help="write the figure (mean ± 1 s.e. over critics)")
    args = parser.parse_args()

    gamma = args.gamma or run_gamma(args.critics[0]) or 0.99
    traj = play_scripted()
    per_critic = np.stack([charges(load_critic(p), traj, gamma, args.delta) for p in args.critics])
    mean = per_critic.mean(axis=0)
    n = len(per_critic)
    sem = per_critic.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)

    print(f"layout seed {LAYOUT_SEED}, {len(mean)} steps, {args.delta} charge, gamma {gamma}, "
          f"mean of {len(args.critics)} critic(s)")
    print("step  " + "  ".join(f"agent {i} (kind)".ljust(34) for i in range(N_AGENTS)))
    for k, row in enumerate(mean):
        cells = [f"{row[i]:+.3f} ({step_kind(traj, k, i)})".ljust(34) for i in range(N_AGENTS)]
        print(f"{k:>4}  " + "  ".join(cells))
    disc = (gamma ** np.arange(len(mean)))[:, None] * mean
    print("discounted total: " + "  ".join(f"agent {i} {disc[:, i].sum():.3f}" for i in range(N_AGENTS)))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["critic", "step", "agent", "kind", "charge"])
            for p, ch in zip(args.critics, per_critic):
                for k in range(len(ch)):
                    for i in range(N_AGENTS):
                        w.writerow([p, k, i, step_kind(traj, k, i), f"{ch[k, i]:.6f}"])
        print(f"wrote {args.csv}")
    if args.pdf:
        plot(traj, mean, sem, args.pdf)
        print(f"wrote {args.pdf}")


if __name__ == "__main__":
    main()
