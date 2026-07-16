import numpy as np
import numpy.typing as npt


def expectile(tau, iters=100):
    """Return a fill callable computing the tau-expectile of an array of values.

    tau=0.5 is the mean; tau -> 1 approaches the max.
    """
    def _fill(values):
        m = values.mean()
        for _ in range(iters):
            w = np.where(values >= m, tau, 1.0 - tau)
            m = np.sum(w * values) / np.sum(w)
        return m
    return _fill


def expectile_fit(tau):
    """Return a fit criterion (residuals -> weights) for FactoredSetEval.learn.

    Asymmetric squared loss rho_tau(u) = |tau - 1{u<0}| * u**2. tau=0.5 is
    ordinary least squares; tau -> 1 fits an upper envelope.
    """
    def _weights(residuals):
        return np.where(residuals >= 0, tau, 1.0 - tau)
    return _weights


def quantile_fit(q, eps=1e-6):
    """Return a fit criterion (residuals -> weights) for FactoredSetEval.learn.

    Pinball loss rho_q(u) = |q - 1{u<0}| * |u|. q=0.5 is the median; q -> 1
    approaches an upper envelope. eps guards the division by |u|.
    """
    def _weights(residuals):
        return np.where(residuals >= 0, q, 1.0 - q) / np.maximum(np.abs(residuals), eps)
    return _weights


class TabularSetEval:
    """The simple set eval function where every entry is a point in a table. The simplest case
    """

    def __init__(self, num_agents : int, action_space_size : int):
        self._n = num_agents
        self._a = action_space_size
        self._set_eval_table = np.full([self._a for _ in range(self._n)], -np.inf)

    def learn(self, dataset : npt.NDArray, fill=np.mean, aggregate=np.mean):
        """Learn the set_eval_table from the data.

        Args:
            dataset (npt.NDArray): An array of shape d x num_agents + 1. The first num_agents
                columns are the action index taken at each agent and the last column is the value
                of the joint action.
            aggregate (Callable[[npt.NDArray], float]): Collapses the repeated samples of one
                seen joint action into its estimate (e.g. np.mean, np.max, expectile(0.99)).
            fill (Callable[[npt.NDArray], float]): Maps the per-joint estimates to the scalar
                placed in every unseen joint action (e.g. np.mean, np.max, expectile(0.99)).
        """

        idx = dataset[:, :-1].astype(int)
        values = dataset[:, -1]

        # Group samples by flat joint index and reduce each group with `aggregate`.
        dims = tuple(self._a for _ in range(self._n))
        flat = np.ravel_multi_index(idx.T, dims)
        order = np.argsort(flat, kind="stable")
        flat_s, values_s = flat[order], values[order]
        bounds = np.flatnonzero(np.diff(flat_s)) + 1
        keys = flat_s[np.concatenate(([0], bounds))]
        est = np.array([aggregate(g) for g in np.split(values_s, bounds)])

        # Unseen joint actions get the fill statistic of the per-joint estimates.
        table = np.full(self._a ** self._n, fill(est))
        table[keys] = est
        self._set_eval_table = table.reshape(dims)


    def predict(self, action_sets : npt.NDArray):
        """Will return the max predicted value within each set of joint actions

        Args:
            action_sets (npt.NDArray): int array of shape (num_sets, set_size,
                num_agents); each row of a set is one joint action.
        """

        values = self._set_eval_table[tuple(np.moveaxis(action_sets, -1, 0))]
        return values.max(axis=-1)


class FactoredSetEval:
    """Set eval under the fully factored assumption: v(a) ~ sum_i f_i(a_i).

    One utility table per agent (num_agents x action_space_size), fit jointly
    on the observed joint actions by reweighted least squares under the chosen
    fit criterion.
    """

    def __init__(self, num_agents : int, action_space_size : int, ridge : float = 1e-3):
        self._n = num_agents
        self._a = action_space_size
        # Ridge resolves the collinearity between the agents' one-hot blocks.
        self._ridge = ridge
        self._f = np.zeros((num_agents, action_space_size))

    def learn(self, dataset : npt.NDArray, fit=None, iters=100):
        """Fit the per-agent utility tables.

        Args:
            dataset (npt.NDArray): same format as TabularSetEval.learn -- the
                first num_agents columns are the action index at each agent and
                the last column is the value of the joint action.
            fit (Callable[[npt.NDArray], npt.NDArray] | None): maps current residuals to
                per-point weights for the next weighted least-squares solve; None means
                ordinary least squares (e.g. expectile_fit(0.99), quantile_fit(0.99)).
            iters (int): max fixed-point iterations of the reweighted solve
                (stops early once the weights settle).
        """
        idx = dataset[:, :-1].astype(int)
        values = dataset[:, -1]
        d = len(values)

        # One-hot design: column i * action_space_size + b is 1 when agent i
        # played action b.
        X = np.zeros((d, self._n * self._a))
        X[np.arange(d)[:, None], idx + self._a * np.arange(self._n)] = 1.0

        #Ridge regression with least squares
        reg = self._ridge * np.eye(self._n * self._a)
        coef = np.linalg.solve(X.T @ X + reg, X.T @ values)
        #If using a fit then it becomes weighted least squares
        if fit is not None:
            w = np.ones(d)
            for _ in range(iters):
                w_new = fit(values - X @ coef)
                if np.allclose(w_new, w, rtol=1e-8):
                    break
                w = w_new
                coef = np.linalg.solve(X.T @ (w[:, None] * X) + reg, X.T @ (w * values))

        self._f = coef.reshape(self._n, self._a)

    def predict(self, action_sets : npt.NDArray):
        """Will return the max predicted value within each set of joint actions

        Each joint action's value is the sum of its agents' individual utilities.

        Args:
            action_sets (npt.NDArray): int array of shape (num_sets, set_size,
                num_agents); each row of a set is one joint action.
        """

        values = self._f[np.arange(self._n), action_sets].sum(axis=-1)
        return values.max(axis=-1)
