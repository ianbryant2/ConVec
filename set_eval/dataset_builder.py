from itertools import combinations

import numpy as np


def pairwise(mix=1.0, pairs=None, rng=None):
    """Return a distribution callable with pairwise interaction structure.

    The value table is a variance-weighted blend of an additive part and a
    pairwise part,

        v(a) = sqrt(1 - mix) * sum_i f_i(a_i) / sqrt(n)
             + sqrt(mix)     * sum_{(i,j) in pairs} g_ij(a_i, a_j) / sqrt(len(pairs)),

    with every f_i entry and g_ij entry drawn i.i.d. standard normal. Each part
    is normalized to unit variance, so `mix` is the fraction of the table's
    variance carried by pairwise interactions: mix=0 is fully factorizable
    (equivalent to additive()), mix=1 is pure pairwise structure.

    Args:
        mix (float): fraction of the table's variance carried by the pairwise
            part (the rest is additive).
        pairs (sequence[tuple[int, int]] | None): which agent pairs interact,
            as (i, j) agent indices. None means every pair.
        rng (np.random.Generator | None): source of the random draws; pass a
            seeded Generator to make the table deterministic. None draws
            from global np.random.

    Returns:
        Callable[[tuple], np.ndarray]: draws a value table of the requested
        shape, so it can be passed straight to RandomDataset(..., distribution=...).
    """
    rng = rng if rng is not None else np.random

    def _sample(size):
        n = len(size)
        chosen = (list(combinations(range(n), 2)) if pairs is None
                  else [tuple(sorted(p)) for p in pairs])
        if any(i == j or not 0 <= i < j < n for i, j in chosen):
            raise ValueError(f"pairs must be distinct agent indices in [0, {n})")
        pair_part = np.zeros(size)
        for i, j in chosen:
            g = rng.standard_normal((size[i], size[j]))
            # Double-center to zero the additive (row/column mean) component,
            # then rescale to restore unit variance.
            g = g - g.mean(axis=0, keepdims=True) - g.mean(axis=1, keepdims=True) + g.mean()
            g *= np.sqrt(size[i] * size[j] / ((size[i] - 1) * (size[j] - 1)))
            shape = [size[k] if k in (i, j) else 1 for k in range(n)]
            pair_part = pair_part + g.reshape(shape)
        pair_part /= np.sqrt(len(chosen))
        return np.sqrt(1 - mix) * additive(rng=rng)(size) + np.sqrt(mix) * pair_part

    return _sample


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
        shape, so it can be passed straight to RandomDataset(..., distribution=...).
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


def multimodal(peaks, spread=1.0, weights=None):
    """Return a distribution callable for a mixture of Gaussians.

    Args:
        peaks (sequence[float]): the mode locations (means of the components).
        spread (float | sequence[float]): std around each peak; a scalar applies
            to every peak, or pass one value per peak.
        weights (sequence[float] | None): relative probability of each peak;
            defaults to equal weight.

    Returns:
        Callable[[tuple], np.ndarray]: draws samples of the requested shape, so it
        can be passed straight to RandomDataset(..., distribution=multimodal(...)).
    """
    peaks = np.asarray(peaks, dtype=float)
    spread = np.asarray(spread, dtype=float)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

    def _sample(size):
        which = np.random.choice(len(peaks), size=size, p=weights)
        centers = peaks[which]
        scale = spread if spread.ndim == 0 else spread[which]
        return np.random.normal(centers, scale)

    return _sample


class RandomDataset:

    def __init__(self, num_agents, num_actions, distribution=None):
        """
        Args:
            num_agents (int): number of agents (dimensions of the value table).
            num_actions (int): number of actions per agent (size of each dimension).
            distribution (Callable[[tuple], np.ndarray]): draws the value table given a
                shape, e.g. lambda size: np.random.normal(0, 1, size). Defaults to
                uniform on [-10, 10].
        """
        self._n = num_agents
        self._a = num_actions
        self._distribution = distribution if distribution is not None else (
            lambda size: np.random.uniform(-10, 10, size=size)
        )

        self.resample()

    def resample(self):
        """Draw a fresh value for every joint action from the distribution."""
        self._data = self._distribution(tuple(self._a for _ in range(self._n)))
        return self._data

    def train_test_split(self, num_train, num_test, num_sets, set_sizes=None):
        """Split the joint-action space into disjoint train and test sets and
        return points ready for a set evaluator, stratified by query-set size.

        Draws `num_train + num_test` distinct joint actions uniformly at random
        from the full |num_actions|**num_agents space and splits them (the two
        sets share no joint action). Then, for each size in `set_sizes`, builds
        `num_sets` random query sets of exactly that size by drawing distinct
        held-out test joint actions.

        Args:
            num_train, num_test: number of joint actions in each split.
            num_sets: number of query sets built per size.
            set_sizes (sequence[int] | None): the query-set sizes; defaults to
                ~10 log-spaced sizes from 1 to num_test.

        Returns:
            train_points: shape (num_train, num_agents + 1). The first num_agents
                columns are the action index at each agent and the last column is
                the value of that joint action -- matches TabularSetEval.learn.
            test_sets: dict mapping set size -> int array of shape
                (num_sets, size, num_agents). Each row of a set is one
                distinct joint action -- matches the action_sets expected
                by predict.
        """
        total = self._a ** self._n
        chosen = np.random.choice(total, size=num_train + num_test, replace=False)
        train_actions = np.stack(np.unravel_index(chosen[:num_train], self._data.shape), axis=1)
        test_actions = np.stack(np.unravel_index(chosen[num_train:], self._data.shape), axis=1)

        # Train points: indices + their value, ready for learn.
        train_values = self._data[tuple(train_actions.T)]
        train_points = np.column_stack([train_actions, train_values])

        # Test sets: random subsets of the held-out test actions, num_sets of
        # them per size.
        n_test = len(test_actions)
        if set_sizes is None:
            set_sizes = np.unique(np.logspace(0, np.log10(n_test), 10).astype(int))

        test_sets = {}
        for k in set_sizes:
            sets = np.empty((num_sets, k, self._n), dtype=int)
            for i in range(num_sets):
                # sample distinct actions so each set has exactly k members
                sets[i] = test_actions[np.random.choice(n_test, size=k, replace=False)]
            test_sets[int(k)] = sets

        return train_points, test_sets

    def error(self, predictions, test_sets, error_func):
        """Error of a set evaluator's predictions against ground truth.

        For each set in `test_sets` (int array of shape (num_sets, set_size,
        num_agents)) looks up the true max value over its joint actions from
        self._data, then combines it with the model's `predictions` via
        `error_func`.
        """
        true_maxima = self._data[tuple(np.moveaxis(test_sets, -1, 0))].max(axis=-1)
        return error_func(predictions, true_maxima)
