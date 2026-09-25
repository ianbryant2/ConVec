"""Normal-form (one-step) games, each isolating one MARL failure mode.

Letters refer to marl_failure_modes_reference.md:

    (A) relative overgeneralization -> risky_coordination, climbing,
        epsilon_greedy_samples
    (B) representational ceiling    -> climbing(num_agents=3), kwise(order>=3)
    (C) equilibrium-selection ties  -> penalty_game, anti_coordination
    (D) scalability                 -> sweep num_agents on any generator
    (F) applicability boundary      -> stag_hunt, decorrelate
    (G) overestimation              -> overestimation_trap
    (H) exploration blindness       -> needle

Rewards live in one tensor of shape (*action_dims, num_agents): the first
num_agents axes index the joint action, the last axis picks the receiving
agent. Common reward is the special case of a constant last axis.
"""

from itertools import combinations

import numpy as np
import numpy.typing as npt


class Game:
    """A one-shot game with per-agent rewards and exact solution diagnostics.

    All diagnostics read the exact reward table; zero-mean observation noise
    lives only in `sample`.
    """

    def __init__(self, rewards: npt.NDArray, noise_scale=0.0):
        """
        Args:
            rewards (npt.NDArray): shape (*action_dims, num_agents); entry
                [a_1, ..., a_n, i] is agent i's reward for joint action a.
            noise_scale (float | npt.NDArray): std of the zero-mean Gaussian
                noise used by `sample`, broadcastable to `rewards.shape`.
                0 (default) makes sampling deterministic.
        """
        self.rewards = np.asarray(rewards, dtype=float)
        # Every axis except the last indexes one agent's action.
        self.num_agents = self.rewards.ndim - 1
        if self.rewards.shape[-1] != self.num_agents:
            raise ValueError(
                f"last axis ({self.rewards.shape[-1]}) must equal the number "
                f"of joint-action axes ({self.num_agents})")
        # Number of actions available to each agent.
        self.action_dims = self.rewards.shape[:-1]
        self.noise_scale = np.asarray(noise_scale, dtype=float)

    @classmethod
    def common(cls, table: npt.NDArray, noise_scale=0.0):
        """Lift a common-reward value table (one axis per agent) to a Game."""
        table = np.asarray(table, dtype=float)
        # Copy the shared value onto a new trailing agent axis (one copy per agent).
        return cls(np.repeat(table[..., None], table.ndim, axis=-1),
                   noise_scale=noise_scale)

    # ------------------------------------------------------------------ payoffs

    def payoff(self, joint_actions: npt.NDArray):
        """Exact rewards for a batch of joint actions.

        Args:
            joint_actions (npt.NDArray): int array (..., num_agents).

        Returns:
            npt.NDArray of shape (..., num_agents): each agent's reward.
        """
        joint_actions = np.asarray(joint_actions)
        # Move the agent axis to the front so it unpacks into an index tuple.
        return self.rewards[tuple(np.moveaxis(joint_actions, -1, 0))]

    def sample(self, joint_actions: npt.NDArray, rng=None):
        """Like `payoff` plus zero-mean Gaussian noise of std `noise_scale`.

        In a common-reward game all agents share one noise draw per joint
        action, so sampled rewards stay common.

        Args:
            rng: numpy random source; defaults to np.random.
        """
        rng = np.random if rng is None else rng
        exact = self.payoff(joint_actions)
        # Expand noise_scale to the full table and look up each joint action's std.
        scale = np.broadcast_to(self.noise_scale, self.rewards.shape)
        scale = scale[tuple(np.moveaxis(np.asarray(joint_actions), -1, 0))]
        # Common reward: one noise draw per joint action, broadcast across
        # agents; otherwise one draw per agent.
        shape = exact.shape[:-1] + ((1,) if self.is_common_reward else (self.num_agents,))
        return exact + scale * rng.standard_normal(shape)

    def common_table(self):
        """The value table of a common-reward game (drops the agent axis)."""
        if not self.is_common_reward:
            raise ValueError("game is not common-reward; use .rewards directly")
        # All agents agree everywhere, so agent 0's slice is the shared table.
        return self.rewards[..., 0]

    def as_distribution(self):
        """A distribution callable (size -> table), so a common-reward game
        can be passed straight to RandomDataset(..., distribution=...)."""
        def _sample(size):
            if tuple(size) != self.action_dims:
                raise ValueError(f"game has action dims {self.action_dims}, "
                                 f"requested {tuple(size)}")
            return self.common_table()
        return _sample

    # -------------------------------------------------------------- diagnostics

    @property
    def is_common_reward(self):
        """True when every agent receives the identical reward everywhere."""
        # ptp (max - min) along the agent axis is 0 iff all agents agree.
        return bool(np.ptp(self.rewards, axis=-1).max() == 0)

    @property
    def is_no_conflict(self):
        """True when some joint action maximizes every agent simultaneously."""
        return len(self.optimal_joints()) > 0

    def optimal_joints(self):
        """Joint actions that give every agent its global-maximum reward.

        Returns:
            int array (num_optima, num_agents); empty for games with conflict.
        """
        mask = np.ones(self.action_dims, dtype=bool)
        for i in range(self.num_agents):
            r = self.rewards[..., i]
            # Keep only cells where agent i attains its global max.
            mask &= r == r.max()
        # Surviving cells maximize everyone; argwhere returns their index rows.
        return np.argwhere(mask)

    def pure_nash(self):
        """All pure-strategy Nash equilibria.

        Returns:
            int array (num_equilibria, num_agents).
        """
        mask = np.ones(self.action_dims, dtype=bool)
        for i in range(self.num_agents):
            r = self.rewards[..., i]
            # Agent i is best-responding where its reward equals the max of
            # its reward slice along its own action axis (partners fixed).
            mask &= r == r.max(axis=i, keepdims=True)
        return np.argwhere(mask)

    def pareto_optima(self, chunk=1024):
        """All Pareto-optimal joint actions.

        Brute force over pairs of joint actions in chunks of size `chunk`.

        Returns:
            int array (num_optima, num_agents).
        """
        # Flatten to one row of per-agent rewards per joint action.
        flat = self.rewards.reshape(-1, self.num_agents)
        dominated = np.zeros(len(flat), dtype=bool)
        for start in range(0, len(flat), chunk):
            block = flat[start:start + chunk]                     # (c, agents)
            # Compare every joint action J against each block member:
            # J dominates it if J is >= for all agents and > for at least one.
            ge_all = (flat[:, None] >= block[None]).all(-1)       # (J, c)
            gt_any = (flat[:, None] > block[None]).any(-1)
            dominated[start:start + chunk] = (ge_all & gt_any).any(axis=0)
        idx = np.flatnonzero(~dominated)
        # Convert flat indices back to joint-action coordinates.
        return np.stack(np.unravel_index(idx, self.action_dims), axis=1)

    def classify(self, joint_actions: npt.NDArray):
        """Label each joint action with the failure mode it exhibits.

        Labels:
            'optimal'          -- an optimal joint action.
            'dominated_nash'   -- a non-optimal pure Nash equilibrium (A).
            'miscoordination'  -- every component belongs to some optimal
                                  joint action but the combination does not (C).
            'other'            -- anything else.

        Args:
            joint_actions (npt.NDArray): int array (..., num_agents).

        Returns:
            object array of labels with shape joint_actions.shape[:-1].
        """
        joint_actions = np.asarray(joint_actions)
        flat = joint_actions.reshape(-1, self.num_agents)
        optima = {tuple(a) for a in self.optimal_joints()}
        nash = {tuple(a) for a in self.pure_nash()}
        # components[i] = agent i's actions that appear in some optimum.
        components = [{a[i] for a in optima} for i in range(self.num_agents)]

        labels = np.empty(len(flat), dtype=object)
        for k, a in enumerate(map(tuple, flat)):
            if a in optima:
                labels[k] = 'optimal'
            elif a in nash:
                labels[k] = 'dominated_nash'
            elif optima and all(a[i] in components[i] for i in range(self.num_agents)):
                # Each agent picked a piece of an optimum, just not the same one.
                labels[k] = 'miscoordination'
            else:
                labels[k] = 'other'
        return labels.reshape(joint_actions.shape[:-1])


# ---------------------------------------------------------- random instances

def additive(rng=None):
    """Return a distribution callable for a fully factorizable value table.

    v(a_1, ..., a_n) = sum_i f_i(a_i), each f_i drawn i.i.d. standard normal,
    scaled by 1/sqrt(n) so the table has unit variance.

    Args:
        rng (np.random.Generator | None): source of the random draws; pass a
            seeded Generator to make the table deterministic. None draws
            from global np.random.

    Returns:
        Callable[[tuple], np.ndarray]: draws a value table of the requested
        shape, e.g. for Game.common(additive()(action_dims)).
    """
    rng = rng if rng is not None else np.random

    def _sample(size):
        n = len(size)
        table = np.zeros(size)
        for i, num_actions in enumerate(size):
            f = rng.standard_normal(num_actions)
            shape = [-1 if j == i else 1 for j in range(n)]
            table = table + f.reshape(shape)
        return table / np.sqrt(n)

    return _sample


def kwise(order=3, mix=1.0, groups=None, num_groups=None, rng=None):
    """Return a distribution callable with pure order-way interaction structure.

    For each group of `order` agents a random tensor is drawn and every
    lower-order (ANOVA) component is projected out by centering along each of
    its axes, leaving coordination of exactly that order (failure (B)).

    Args:
        mix (float): fraction of the table's variance carried by the
            order-way part (the rest is additive, via additive()).
        order (int): how many agents each interaction couples.
        groups (sequence[tuple[int, ...]] | None): which agent groups
            interact, as tuples of `order` distinct agent indices. None means
            every group of that size.
        num_groups (int | None): draw this many distinct groups uniformly at
            random instead of using all of them; the draw is constrained to
            cover every agent. Mutually exclusive with `groups`.
        rng (np.random.Generator | None): source of the random draws; pass a
            seeded Generator to make the table deterministic. None draws
            from global np.random.

    Returns:
        Callable[[tuple], np.ndarray]: draws a value table of the requested
        shape, ready to pass to Game.common.
    """
    if groups is not None and num_groups is not None:
        raise ValueError("pass either groups or num_groups, not both")
    rng = rng if rng is not None else np.random

    def _sample(size):
        n = len(size)
        if num_groups is not None:
            all_groups = list(combinations(range(n), order))
            min_needed = -(-n // order)  # ceil: fewer groups cannot cover all agents
            if not min_needed <= num_groups <= len(all_groups):
                raise ValueError(
                    f"num_groups must be in [{min_needed}, {len(all_groups)}] to "
                    f"cover {n} agents with distinct groups of {order}")
            while True:  # rejection-sample until every agent is in some group
                chosen = [all_groups[i] for i in
                          rng.choice(len(all_groups), size=num_groups, replace=False)]
                if len({i for g in chosen for i in g}) == n:
                    break
        else:
            chosen = (list(combinations(range(n), order)) if groups is None
                      else [tuple(sorted(g)) for g in groups])
            if any(len(set(g)) != order or not all(0 <= i < n for i in g) for g in chosen):
                raise ValueError(f"groups must be {order} distinct agent indices in [0, {n})")
        part = np.zeros(size)
        for g in chosen:
            t = rng.standard_normal(tuple(size[i] for i in g))
            # Center along each axis to zero every lower-order ANOVA term,
            # then rescale to restore unit variance.
            for ax in range(order):
                t = t - t.mean(axis=ax, keepdims=True)
            t *= np.sqrt(np.prod([size[i] for i in g])
                         / np.prod([size[i] - 1 for i in g]))
            shape = [size[k] if k in g else 1 for k in range(n)]
            part = part + t.reshape(shape)
        part /= np.sqrt(len(chosen))
        return np.sqrt(1 - mix) * additive(rng=rng)(size) + np.sqrt(mix) * part

    return _sample


# ------------------------------------------------------------------- generators

def budget_table1():
    """Three-agent general-sum game from Table 1 of the budget writeup."""
    # Axes: agent 1 action, agent 2 action, agent 3 action, recipient.
    # 维度依次为三个智能体的动作和奖励接收者；A/B/C 对应 0/1/2。
    rewards = np.zeros((3, 3, 3, 3), dtype=float)

    # Agent 3 plays A. Rows: agent 1; columns: agent 2.
    # 智能体 3 选择 A；行对应智能体 1，列对应智能体 2。
    rewards[:, :, 0, :] = [
        [[10, 10, 10], [8, 16, 7], [0, 0, 0]],
        [[15, 8, 7],   [9, 9, 9],  [0, 0, 0]],
        [[0, 0, 0],    [0, 0, 0],  [16, 0, 0]],
    ]

    # Agent 3 plays B / 智能体 3 选择 B。
    rewards[:, :, 1, :] = [
        [[8, 7, 14], [0, 0, 0],  [0, 0, 0]],
        [[0, 0, 0],  [7, 7, 9],  [2, 15, 1]],
        [[0, 0, 0],  [15, 2, 1], [0, 0, 0]],
    ]

    # Agent 3 plays C / 智能体 3 选择 C。
    rewards[:, :, 2, :] = [
        [[0, 0, 0], [0, 0, 0],  [0, 0, 0]],
        [[0, 0, 0], [2, 1, 15], [3, 2, 2]],
        [[0, 0, 0], [2, 3, 2],  [2, 2, 2]],
    ]

    return Game(rewards)


def risky_coordination(num_agents=2, num_actions=3, reward=8.0, penalty=-12.0):
    """Failure (A): the matrix (a)/(b)/(c) family, generalized to N agents.

    Common reward: `reward` when all agents play action 0, `penalty` when
    some but not all do, 0 otherwise. penalty=0 is matrix (a), (8, -12) is
    matrix (b), reward=4 gives matrix (c).
    """
    if reward <= 0:
        raise ValueError("reward must be > 0 so the risky joint action stays optimal")
    shape = tuple(num_actions for _ in range(num_agents))
    # Per joint action, count how many agents chose the risky action 0.
    count0 = (np.indices(shape) == 0).sum(axis=0)
    table = np.where(count0 == num_agents, reward,          # everyone committed
                     np.where(count0 > 0, penalty, 0.0))    # partial commit / nobody
    return Game.common(table)


def climbing(num_agents=2):
    """Failures (A) and, at num_agents >= 3, (B).

    The Claus & Boutilier climbing game generalized to num_agents >= 2,
    matching the 2- and 3-agent tables of Christianos et al. (Pareto-AC,
    Figs. 6 and 7) cell for cell:

        all play 0                           -> 11
        exactly one agent deviates to 1      -> -30
        one of the first num_agents-2 agents
          deviates alone to 2                -> -30  (absent at num_agents=2)
        all play 1                           -> 7
        last agent plays 1, rest play 2      -> 6
        all play 2                           -> 5
        anything else                        -> 0

    Note: the uoe-agents/matrix-games repo omits the lone-deviation-to-2
    cliff that appears in the paper's Fig. 7; this follows the figure.
    """
    shape = tuple(3 for _ in range(num_agents))
    idx = np.indices(shape)
    count_a, count_b = (idx == 0).sum(axis=0), (idx == 1).sum(axis=0)
    count_c = (idx == 2).sum(axis=0)
    all_a = count_a == num_agents
    all_b = count_b == num_agents
    all_c = count_c == num_agents
    one_dev_to_b = (count_b == 1) & (count_a == num_agents - 1)
    # Lone deviation to action 2 by any of the first num_agents - 2 agents
    # (empty slice -> all False at num_agents=2, keeping the 2-agent matrix).
    early_dev_to_c = ((count_c == 1) & (count_a == num_agents - 1)
                      & (idx[:num_agents - 2] == 2).any(axis=0))
    last_b_rest_c = (idx[-1] == 1) & (count_c == num_agents - 1)
    table = np.select(
        [all_a, one_dev_to_b, early_dev_to_c, all_b, last_b_rest_c, all_c],
        [11., -30., -30., 7., 6., 5.],
        default=0.)
    return Game.common(table)


def penalty_game(num_agents=2, num_coord=2, reward=10.0, cliff=-100.0, safe=2.0):
    """Failure (C): tied Pareto-optimal equilibria with cliffs between them.

    Actions 0..num_coord-1 coordinate, the last action is safe. The
    `num_coord` optima are cyclic shifts (agent i plays (j + i) mod
    num_coord). Other all-coordination joints hit the `cliff`; everyone-safe
    earns `safe`; mixed safe/coordination earns 0. The defaults reproduce
    the classic Penalty game up to relabeling.
    """
    if not (safe < reward and 0.0 < reward):
        raise ValueError("need safe < reward and reward > 0 to keep the shifts optimal")
    num_actions = num_coord + 1
    shape = tuple(num_actions for _ in range(num_agents))
    idx = np.indices(shape)

    all_coord = (idx < num_coord).all(axis=0)     # nobody played the safe action
    all_safe = (idx == num_coord).all(axis=0)     # everybody played it
    # Mark the num_coord winning patterns: shift j means agent i plays (j+i) mod num_coord.
    on_shift = np.zeros(shape, dtype=bool)
    for j in range(num_coord):
        pattern = (np.arange(num_agents) + j) % num_coord
        # Reshape so pattern[i] compares against agent i's axis of idx.
        on_shift |= (idx == pattern.reshape(-1, *([1] * num_agents))).all(axis=0)

    table = np.select([all_coord & on_shift, all_coord, all_safe],
                      [reward, cliff, safe], default=0.0)
    return Game.common(table)


def anti_coordination(num_agents=2, num_actions=None, reward=10.0, penalty=-10.0):
    """Failure (C) with a large number of tied optima.

    Common reward: `reward` when all agents play *distinct* actions,
    `penalty` on any collision. With a actions and n agents there are
    a!/(a-n)! tied optima.
    """
    if num_actions is None:
        num_actions = num_agents
    if num_actions < num_agents:
        raise ValueError("need num_actions >= num_agents for a collision-free joint")
    shape = tuple(num_actions for _ in range(num_agents))
    # Columns of idx.T are joint actions; distinct components = no collision.
    idx = np.indices(shape).reshape(num_agents, -1)
    distinct = np.array([len(set(col)) == num_agents for col in idx.T])
    table = np.where(distinct.reshape(shape), reward, penalty)
    return Game.common(table)


def stag_hunt(num_agents=2, stag=4.0, hare_tempt=3.0, hare_safe=2.0, sucker=0.0):
    """Failure (F): no-conflict but *not* common-reward.

    Two actions, 0 = stag, 1 = hare, with per-agent rewards:

        play stag, everyone does   -> stag (the shared optimum)
        play stag, someone defects -> sucker
        play hare, a partner still
          hunts stag               -> hare_tempt
        play hare, nobody hunts    -> hare_safe

    Requires hare_tempt < stag to stay no-conflict.
    """
    if not hare_tempt < stag:
        raise ValueError("hare_tempt must be < stag to keep the game no-conflict")
    shape = tuple(2 for _ in range(num_agents))
    idx = np.indices(shape)
    plays_stag = idx == 0                       # (num_agents, *shape)
    total_stag = plays_stag.sum(axis=0)         # stag hunters per joint action

    # Rewards differ per agent, so fill the tensor one agent slice at a time.
    rewards = np.empty(shape + (num_agents,))
    for i in range(num_agents):
        # Does at least one *other* agent hunt stag (excluding agent i itself)?
        others_hunt = (total_stag - plays_stag[i]) > 0
        rewards[..., i] = np.where(
            plays_stag[i],
            np.where(total_stag == num_agents, stag, sucker),
            np.where(others_hunt, hare_tempt, hare_safe))
    return Game(rewards)


def decorrelate(game: Game, rho: float, rng=None):
    """Failure (F) as a spectrum: common reward -> idiosyncratic rewards.

    Replaces each agent's off-optimum reward with the blend
    rho * v + sqrt(1 - rho^2) * sigma * eps_i (eps_i i.i.d. standard normal
    per agent, sigma the table's std). rho=1 recovers the common-reward
    game; rho=0 makes off-optimum rewards pure per-agent noise. Optimal
    entries are copied through exactly and blended entries are clipped just
    below the optimum, so the game stays no-conflict with the same optima.
    """
    rng = np.random if rng is None else rng
    v = game.common_table()
    sigma = v.std()
    # One independent noise draw per (joint action, agent) pair.
    eps = rng.standard_normal(v.shape + (game.num_agents,))
    blended = rho * v[..., None] + np.sqrt(1.0 - rho ** 2) * sigma * eps

    # Clip blended values just below the optimum so no off-optimum entry can
    # tie or beat it; margin scales with the table's range.
    margin = 1e-6 * (np.ptp(v) + 1.0)
    blended = np.minimum(blended, v.max() - margin)
    # Optimal cells keep their exact common value; everything else is blended.
    optima_mask = v == v.max()
    rewards = np.where(optima_mask[..., None], v[..., None], blended)
    return Game(rewards)


def needle(num_agents=2, num_actions=3, reward=10.0, penalty=-5.0):
    """Failure (H): a sparse optimum surrounded by cliffs, with an opt-out.

    Common reward: the single needle (all agents play action 0) is worth
    `reward`; each agent's last action is a null action making the joint
    worth 0 whenever anyone plays it; everything else costs `penalty`.
    """
    if reward <= 0:
        raise ValueError("reward must be > 0 so the needle beats opting out")
    shape = tuple(num_actions for _ in range(num_agents))
    idx = np.indices(shape)
    is_needle = (idx == 0).all(axis=0)               # everyone on action 0
    any_null = (idx == num_actions - 1).any(axis=0)  # someone opted out
    table = np.where(is_needle, reward, np.where(any_null, 0.0, penalty))
    return Game.common(table)


def overestimation_trap(num_agents=2, num_actions=3, optimum=1.0, gap=1.0,
                        sigma=3.0, frac_noisy=0.5, rng=None):
    """Failure (G): optimistic estimators inflate high-variance mediocrity.

    Common reward with stochastic observations: the true optimum (all
    agents play action 0) has mean `optimum` and zero noise; every other
    joint action has mean `optimum - gap`, and a random `frac_noisy`
    fraction of them carries zero-mean noise of std `sigma`.
    """
    if gap <= 0:
        raise ValueError("gap must be > 0 so the true optimum stays strictly optimal")
    rng = np.random if rng is None else rng
    shape = tuple(num_actions for _ in range(num_agents))
    is_opt = (np.indices(shape) == 0).all(axis=0)    # the all-0 joint action
    table = np.where(is_opt, optimum, optimum - gap)
    # Pick a random frac_noisy subset of the non-optimal cells to be noisy.
    noisy = (rng.random(shape) < frac_noisy) & ~is_opt
    scale = np.where(noisy, sigma, 0.0)
    # Trailing axis broadcasts the per-cell std to all agents.
    return Game.common(table, noise_scale=scale[..., None])


# --------------------------------------------------------------------- sampling

def epsilon_greedy_samples(game: Game, epsilon: float, num_samples: int,
                           greedy_target=None, rng=None):
    """Sample joint actions from a frozen epsilon-greedy joint policy.

    Each agent independently plays its component of `greedy_target` with
    probability 1 - epsilon and a uniform-random action otherwise.

    Args:
        game (Game): the game to sample from.
        epsilon (float): per-agent exploration probability.
        num_samples (int): joint actions drawn, with repetition.
        greedy_target (sequence[int] | None): the joint action agents are
            greedy toward; defaults to the game's first optimal joint action.
        rng: numpy random source; defaults to np.random.

    Returns:
        For a common-reward game, an array (num_samples, num_agents + 1)
        matching TabularSetEval.learn / FactoredSetEval.learn: action
        indices then the (possibly noisy) sampled value. For a general game
        the last num_agents columns hold each agent's sampled reward.
    """
    rng = np.random if rng is None else rng
    if greedy_target is None:
        optima = game.optimal_joints()
        if len(optima) == 0:
            raise ValueError("game has no conflict-free optimum; pass greedy_target explicitly")
        greedy_target = optima[0]
    greedy_target = np.asarray(greedy_target)

    # Each agent explores independently with probability epsilon.
    explore = rng.random((num_samples, game.num_agents)) < epsilon
    random_actions = np.stack(
        [rng.randint(0, d, size=num_samples) for d in game.action_dims], axis=1)
    # Exploring agents take their random action, the rest stay on target.
    joints = np.where(explore, random_actions, greedy_target)
    values = game.sample(joints)
    if game.is_common_reward:
        # All agents' samples are identical; keep a single value column.
        values = values[:, :1]
    return np.column_stack([joints, values])
