"""Is the LBF budget c feasible for the intended strategy?

Env lbf_envs:Foraging-5x5-3p-2f-budget-v0 (3 level-1 agents, 2 apples paying
1/2 each, split among the agents loading it). Intended outcome: agent 0 eats
one apple alone, agents 1 and 2 load the other together. Scripted agents play
it, once with agent 0 on each apple, and each agent's discounted concession
is measured two ways:

exact    LBF moves are deterministic, so the exact per-step charges telescope:
             sum_t gamma^t delta_i,t = V*_i(s_0) - G_i
         with V*_i(s_0) the most agent i could get controlling every agent
         (it eats both apples alone, as fast as possible) and G_i its
         discounted return. V*_i treats only apples as obstacles (the other
         agents can always step aside under joint control).
learned  an additive critic (optimq_additive_linear) trained offline on
         random play with a sacred run's concession settings, as in the
         explore_only runs, then charged on the same scripted episodes as
         the budget runner would (target nets, gamma^t delta_t).

The budget is feasible for an episode when some assignment keeps every agent
within c.

    python scripts/lbf_feasibility.py --episodes 500                      # exact only
    python scripts/lbf_feasibility.py --episodes 500 --critic-steps 1500000  # + learned
"""
import argparse
import itertools
import json
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PROJECT / "epymarl" / "src"), str(PROJECT)]

from envs.gymma import GymmaWrapper  # noqa: E402

KEY = "lbf_envs:Foraging-5x5-3p-2f-budget-v0"
TIME_LIMIT = 25
N_AGENTS, N_ACTIONS = 3, 6
NONE, NORTH, SOUTH, WEST, EAST, LOAD = range(6)
MOVES = {NORTH: (-1, 0), SOUTH: (1, 0), WEST: (0, -1), EAST: (0, 1)}
C = np.array([0.55, 0.8, 0.8])
# A sacred run whose concession settings the learned critic copies: PAC,
# potential, explore_only, additive, seed 0 (scripts/sweeps lbf_budget probe).
REFERENCE_RUN = PROJECT / "epymarl/results/sacred/pac_sarsa_ns/lbf_envs:Foraging-5x5-3p-2f-budget-v0/20"


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


def utopian_value(pos, foods, gamma, rows, cols, n_spawned=None, steps_left=TIME_LIMIT):
    """V*_i(s): agent i eats every remaining apple alone, as early as
    possible. Each apple pays 1 / (apples spawned) to a lone loader."""
    foods = list(foods)
    if not foods or steps_left <= 0:
        return 0.0
    best, pay = 0.0, 1.0 / (n_spawned or len(foods))
    for order in ([0, 1], [1, 0]) if len(foods) == 2 else ([0],):
        first, rest = foods[order[0]], [foods[i] for i in order[1:]]
        for cell in adjacent_cells(first, set(foods), rows, cols):
            d1, _ = bfs(pos, {cell}, set(foods), rows, cols)
            if d1 is None or d1 > steps_left - 1:
                continue
            value = pay * gamma**d1  # loads d1 steps from now
            if rest:
                second = rest[0]
                d2, _ = bfs(cell, set(adjacent_cells(second, {second}, rows, cols)), {second}, rows, cols)
                t2 = d1 + 1 + d2 if d2 is not None else None
                if t2 is not None and t2 <= steps_left - 1:
                    value += pay * gamma**t2
            best = max(best, value)
    return best


def assignments(n_foods):
    """Who loads which apple: [{apple index: [agents]}]. Two apples: the
    intended split, with agent 0 on either apple. One apple (some layouts
    spawn only one): agent 0 alone, agents 1 and 2, or all three."""
    if n_foods == 2:
        return [{0: [0], 1: [1, 2]}, {1: [0], 0: [1, 2]}]
    return [{0: [0]}, {0: [1, 2]}, {0: [0, 1, 2]}]


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


def play_scripted(seed, assignment, gamma):
    """One scripted episode. Returns its exact concessions and the trajectory."""
    env = make_env(seed)
    env.reset(seed=seed)
    g = game(env)
    rows, cols = g.field.shape
    foods = [tuple(f) for f in np.argwhere(g.field > 0)]
    v_star = np.array([utopian_value(tuple(p.position), foods, gamma, rows, cols) for p in g.players])
    policy = Scripted(env, assignment)
    returns = np.zeros(N_AGENTS)
    rewards_total = np.zeros(N_AGENTS)
    # Per step: the critic's inputs, plus what exact per-step deltas need
    # (positions and apples before the step, rewards of the step).
    traj = {"state": [], "actions": [], "t": [], "pos": [], "foods": [], "reward": [],
            "n_spawned": len(foods), "rows": rows, "cols": cols}
    for t in range(TIME_LIMIT):
        actions = policy.act(env)
        traj["state"].append(env.get_state())
        traj["actions"].append(actions)
        traj["t"].append(t)
        traj["pos"].append([tuple(p.position) for p in g.players])
        traj["foods"].append([tuple(f) for f in np.argwhere(g.field > 0)])
        _, reward, done, truncated, _ = env.step(actions)
        traj["reward"].append(list(reward))
        returns += gamma**t * np.asarray(reward)
        rewards_total += reward
        if done or truncated:
            break
    traj["pos"].append([tuple(p.position) for p in g.players])
    traj["foods"].append([tuple(f) for f in np.argwhere(g.field > 0)])
    traj["done"] = bool(done)
    env.close()
    planned = np.zeros(N_AGENTS)
    for agents in assignment.values():
        planned[agents] = 1.0 / len(foods) / len(agents)
    return {"exact": v_star - returns, "v_star": v_star, "return": returns,
            "as_planned": np.allclose(rewards_total, planned), "length": t + 1, "traj": traj}


# --- learned critic ----------------------------------------------------------

def random_episodes(env, rng, n):
    """Transitions of n uniformly random episodes, as the runner records them."""
    rows = {k: [] for k in ("state", "t", "actions", "env_reward", "next_state", "next_avail", "terminated")}
    for _ in range(n):
        env.reset(seed=int(rng.integers(2**31)))
        state = env.get_state()
        for t in range(TIME_LIMIT):
            actions = rng.integers(N_ACTIONS, size=N_AGENTS)
            _, reward, done, truncated, _ = env.step(actions)
            nxt = env.get_state()
            last = done or truncated or t == TIME_LIMIT - 1
            rows["state"].append(state)
            rows["t"].append(t)
            rows["actions"].append(actions)
            rows["env_reward"].append(reward)
            rows["next_state"].append(nxt)
            rows["next_avail"].append(np.ones((N_AGENTS, N_ACTIONS)))
            rows["terminated"].append([float(last)])
            state = nxt
            if last:
                break
    return rows


def train_critic(total_steps, seed, run_dir=REFERENCE_RUN, log_every=100_000, on_log=None, replay=None,
                 target=None):
    """on_log(model, steps) is called at each log point, e.g. to evaluate.
    replay overrides the run's replay settings, e.g. {"prioritized": True,
    "alpha": 0.6, "beta": 0.4}; target its TD target ("double_self" or
    "double_cross")."""
    import torch as th
    from components.optimq_additive import OptimQAdditiveLinear

    config = json.loads((Path(run_dir) / "config.json").read_text())
    budget = config["budget"]
    options = dict(budget["concession_args"][budget["concession"]])
    replay_ratio = options.pop("training")["replay_ratio"]
    options.pop("data")
    if replay is not None:
        options["replay"] = replay
    if target is not None:
        options["target"] = target
    if budget["concession"] != "optimq_additive_linear":
        raise ValueError(f"{run_dir} used concession '{budget['concession']}', expected the additive critic")
    args = SimpleNamespace(n_agents=N_AGENTS, n_actions=N_ACTIONS, gamma=config["gamma"],
                           device="cpu", env_args={"time_limit": TIME_LIMIT})
    th.manual_seed(seed)
    env = make_env(seed)
    env.reset(seed=seed)
    model = OptimQAdditiveLinear(args, len(env.get_state()), **options)
    print(f"critic settings from {Path(run_dir).name}: {options}, replay_ratio {replay_ratio}")

    rng = np.random.default_rng(seed)
    seen, credit, next_log, start = 0, 0.0, log_every, time.time()
    stats = {}
    while seen < total_steps:
        rows = random_episodes(env, rng, 10)  # one parallel-runner rollout
        dtype = {"actions": th.long}
        batch = {k: th.as_tensor(np.asarray(v), dtype=dtype.get(k, th.float32)) for k, v in rows.items()}
        model.add(batch)
        seen += len(rows["state"])
        credit += replay_ratio * len(rows["state"])
        n_steps, credit = int(credit), credit - int(credit)
        stats = model.train(n_steps, batch) or stats
        if seen >= next_log:
            print(f"  {seen:>9,} random steps  loss {stats.get('concession_loss', float('nan')):.4f}  "
                  f"critic_gap {stats.get('concession_critic_gap', float('nan')):.3f}  "
                  f"({time.time() - start:.0f}s)", flush=True)
            if on_log is not None:
                on_log(model, seen)
            next_log += log_every
    env.close()
    return model


def learned_concessions(model, traj, gamma):
    import torch as th
    charged = np.zeros(N_AGENTS)
    avail = th.ones(1, N_AGENTS, N_ACTIONS)
    for state, actions, t in zip(traj["state"], traj["actions"], traj["t"]):
        delta = model(th.as_tensor(state[None], dtype=th.float32),
                      th.as_tensor([actions]), avail, t)
        charged += gamma**t * delta[0].numpy()
    return charged


STEP_KINDS = ["move", "stay", "own load, first apple", "other's load, first apple",
              "own load, last apple", "other's load, last apple"]


def step_kind(traj, k, i):
    """What step k of a scripted trajectory is for agent i. Loads are split by
    whether they eat the episode's last apple: that step is terminal, so its
    TD target is just the reward."""
    eaten = len(traj["foods"][k]) > len(traj["foods"][k + 1])
    if traj["reward"][k][i] > 0 or eaten:
        who = "own load" if traj["reward"][k][i] > 0 else "other's load"
        return f"{who}, {'last' if not traj['foods'][k + 1] else 'first'} apple"
    return "move" if traj["pos"][k + 1][i] != traj["pos"][k][i] else "stay"


def per_step(model, results, gamma):
    """Learned vs exact delta at every step of scripted episodes, split into
    delta's two parts: the best joint action's value (max_a Q) and the taken
    one's (Q(s, a_t)). Exact: V*_i(s_t) and r_i,t + gamma V*_i(s_t+1).
    Returns one dict per (step, agent)."""
    import torch as th
    rows = []
    for r in results:
        tr = traj = r["traj"]
        n = len(traj["state"])
        states = th.as_tensor(np.asarray(traj["state"]), dtype=th.float32)
        t = th.as_tensor(traj["t"])
        actions = th.as_tensor(np.asarray(traj["actions"]))
        with th.no_grad():
            inputs = model._inputs(states, t)
            taken = model._joint_index(actions)
            q1, q2 = (c.q(c.target_net, inputs) for c in model.critics)
            best1, best2 = q1.argmax(-1, keepdim=True), q2.argmax(-1, keepdim=True)
            max_part = 0.5 * (q2.gather(-1, best1) + q1.gather(-1, best2)).squeeze(-1)
            taken_part = 0.5 * (q2.gather(-1, taken) + q1.gather(-1, taken)).squeeze(-1)
        v = lambda k, i: utopian_value(tr["pos"][k][i], tr["foods"][k], gamma, tr["rows"], tr["cols"],
                                       tr["n_spawned"], TIME_LIMIT - k)
        for k in range(n):
            for i in range(N_AGENTS):
                v_now = v(k, i)
                q_taken = tr["reward"][k][i] + gamma * (0.0 if (tr["done"] and k == n - 1) else v(k + 1, i))
                rows.append({"kind": step_kind(tr, k, i), "agent": i, "t": k,
                             "exact": v_now - q_taken, "learned": float(max_part[k, i] - taken_part[k, i]),
                             "exact_v": v_now, "learned_max": float(max_part[k, i]),
                             "exact_q": q_taken, "learned_q": float(taken_part[k, i])})
    return rows


def td_errors(model, results, gamma):
    """Per (step, agent) of scripted episodes: the TD error the critic's own
    training target gives (online net selects, target net evaluates, as in
    OptimQ._train_step), next to its error against the exact Q*. Averaged
    over the two critics."""
    import torch as th
    rows = []
    for r in results:
        tr = r["traj"]
        n = len(tr["state"])
        states = th.as_tensor(np.asarray(tr["state"]), dtype=th.float32)
        # The step after the last one is terminal when the episode ended; its
        # next-state input is then masked out, so any state will do.
        nxt_states = th.cat([states[1:], states[-1:]])
        t = th.arange(n)
        taken = model._joint_index(th.as_tensor(np.asarray(tr["actions"])))
        reward = th.as_tensor(np.asarray(tr["reward"]), dtype=th.float32)
        alive = th.ones(n, 1)
        alive[-1] = 0.0 if tr["done"] else 1.0
        td = q_taken = 0.0
        with th.no_grad():
            inputs, nxt = model._inputs(states, t), model._inputs(nxt_states, t + 1)
            for c in model.critics:
                q = c.q_at(c.net, inputs, taken).squeeze(-1)
                best = c.q(c.net, nxt).argmax(-1, keepdim=True)
                target = reward + gamma * alive * c.q_at(c.target_net, nxt, best).squeeze(-1)
                td = td + (q - target) / len(model.critics)
                q_taken = q_taken + q / len(model.critics)
        v = lambda k, i: utopian_value(tr["pos"][k][i], tr["foods"][k], gamma, tr["rows"], tr["cols"],
                                       tr["n_spawned"], TIME_LIMIT - k)
        for k in range(n):
            for i in range(N_AGENTS):
                q_star = tr["reward"][k][i] + gamma * (0.0 if (tr["done"] and k == n - 1) else v(k + 1, i))
                rows.append({"kind": step_kind(tr, k, i), "agent": i, "t": k, "td_error": float(td[k, i]),
                             "error_vs_q_star": float(q_taken[k, i]) - q_star})
    return rows


def report_per_step(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    df["err"] = df.learned - df.exact
    df["max_err"] = df.learned_max - df.exact_v
    df["q_err"] = df.learned_q - df.exact_q
    cols = ["exact", "learned", "err", "max_err", "q_err"]
    print("\n== per step: learned vs exact delta (undiscounted), two-apple scripted episodes")
    print("err = learned - exact = max_err - q_err; max_err = learned max_a Q - V*, q_err = learned Q(s,a_t) - Q*")
    by_kind = df.groupby("kind")[cols].mean().join(df.groupby("kind").size().rename("steps"))
    print(by_kind.round(3).to_string())
    print("\nby agent:")
    print(df.groupby("agent")[cols].mean().round(3).to_string())
    print("\nby step index:")
    print(df[df.t < 8].groupby("t")[cols].mean().join(df.groupby("t").size().rename("rows")).round(3).to_string())


# --- report ------------------------------------------------------------------

def _label(assignment):
    return ", ".join(f"apple {k}: {'+'.join(map(str, a))}" for k, a in assignment.items())


def summarise(name, conc, labels, c=C):
    """conc: (episodes, assignments, agents). Per assignment and the best one
    per episode (the one furthest within, or least over, budget)."""
    worst = (conc - c).max(-1)  # (episodes, assignments); > 0 means someone is over
    best = conc[np.arange(len(conc)), worst.argmin(-1)]
    print(f"\n== {name} concessions (discounted), c = {c.tolist()}")
    print(f"{'':26s}{'agent 0':>16s}{'agent 1':>16s}{'agent 2':>16s}{'within c':>10s}")
    for label, x in list(zip(labels, conc.transpose(1, 0, 2))) + [("best per episode", best)]:
        cells = "".join(f"{x[:, i].mean():7.3f} ±{x[:, i].std():5.3f}  " for i in range(N_AGENTS))
        print(f"{label:26s}{cells}{(x <= c).all(-1).mean():9.1%}")
    q = np.percentile(best, [10, 50, 90], axis=0)
    print("best per episode, 10/50/90th percentiles: "
          + "  ".join(f"agent {i}: {q[0, i]:.2f}/{q[1, i]:.2f}/{q[2, i]:.2f}" for i in range(N_AGENTS)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--critic-steps", type=int, default=0,
                        help="random env steps to train the learned critic on (0: exact only)")
    parser.add_argument("--run", type=Path, default=REFERENCE_RUN,
                        help="sacred run dir whose concession settings the critic copies")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=250_000,
                        help="random steps between critic progress reports")
    parser.add_argument("--save", type=Path, help="save the trained critic here")
    parser.add_argument("--load", type=Path, help="use a critic saved with --save instead of training")
    parser.add_argument("--per", action="store_true",
                        help="train the critic with prioritised replay (overrides the run's setting)")
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta", type=float, default=0.4)
    parser.add_argument("--target", choices=["double_self", "double_cross"],
                        help="the critic's TD target (overrides the run's setting)")
    parser.add_argument("--per-step", action="store_true",
                        help="break learned vs exact delta down by step (needs a critic)")
    args = parser.parse_args()

    gamma = json.loads((args.run / "config.json").read_text())["gamma"]
    # Layouts by apple count: {n_foods: [[result per assignment] per layout]}
    layouts = {2: [], 1: []}
    for e in range(args.episodes):
        seed = 10_000 + e
        env = make_env(seed)
        env.reset(seed=seed)
        n_foods = int((game(env).field > 0).sum())
        env.close()
        layouts[n_foods].append([play_scripted(seed, a, gamma) for a in assignments(n_foods)])
    print(f"{args.episodes} layouts, gamma {gamma}: {len(layouts[2])} with two apples, "
          f"{len(layouts[1])} with one (it then pays 1, not 1/2)")

    model = None
    if args.load:
        import torch as th
        model = th.load(args.load, weights_only=False)
        print(f"\nloaded critic from {args.load}")
    elif args.critic_steps:
        # Progress: learned charges on two-apple layouts, best assignment per episode.
        probe = layouts[2][:100]
        exact_probe = np.array([[r["exact"] for r in ep] for ep in probe])

        def progress(model, steps):
            learned = np.array([[learned_concessions(model, r["traj"], gamma) for r in ep] for ep in probe])
            best = learned[np.arange(len(learned)), (learned - C).max(-1).argmin(-1)]
            err = (learned - exact_probe).mean((0, 1))
            print(f"      two-apple learned charge (best assignment) {best.mean(0).round(3).tolist()}, "
                  f"within c {(best <= C).all(-1).mean():.0%}; learned - exact {err.round(3).tolist()}",
                  flush=True)

        print(f"\ntraining the additive critic on {args.critic_steps:,} random steps")
        replay = ({"prioritized": True, "alpha": args.per_alpha, "beta": args.per_beta}
                  if args.per else None)
        model = train_critic(args.critic_steps, args.seed, args.run, log_every=args.log_every,
                             on_log=progress, replay=replay, target=args.target)
        if args.save:
            import torch as th
            th.save(model, args.save)
            print(f"saved critic to {args.save}")

    for n_foods, eps in layouts.items():
        if not eps:
            continue
        labels = [_label(a) for a in assignments(n_foods)]
        planned = np.mean([[r["as_planned"] for r in ep] for ep in eps], axis=0)
        print(f"\n######## {len(eps)} layouts with {n_foods} apple(s)")
        print("scripted play went as planned: "
              + ", ".join(f"{label} {p:.0%}" for label, p in zip(labels, planned))
              + f"; mean length {np.mean([[r['length'] for r in ep] for ep in eps]):.1f} steps")
        v_star = np.array([ep[0]["v_star"] for ep in eps])
        print(f"V*_i(s_0) mean: {v_star.mean(0).round(3).tolist()}")
        exact = np.array([[r["exact"] for r in ep] for ep in eps])
        summarise("EXACT", exact, labels)
        if model is not None:
            learned = np.array([[learned_concessions(model, r["traj"], gamma) for r in ep] for ep in eps])
            summarise("LEARNED", learned, labels)
            err = learned - exact
            print(f"learned - exact, per agent: mean {err.mean((0, 1)).round(3).tolist()}, "
                  f"std {err.std((0, 1)).round(3).tolist()}")
            if n_foods == 2 and args.per_step:
                report_per_step(per_step(model, [r for ep in eps for r in ep], gamma))


if __name__ == "__main__":
    main()
