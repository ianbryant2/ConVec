"""Budget reward rules: turn the budget state after a step into rewards.

Each rule is registered with the reward setting it produces:
  common=True   returns one float shared by every agent (common_reward=True)
  common=False  returns one reward per agent            (common_reward=False)

main.py's config hook sets common_reward from budget.rule, so choosing
the rule is enough. BudgetWrapper re-checks the pairing, since an explicit
common_reward=... on the command line overrides the hook.

To add a rule:

    @budget_rule("my_rule", common=False)
    def my_rule(s: BudgetStep):
        return ...  # array of shape (n_agents,)
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BudgetStep:
    """Everything a rule may depend on after one environment step."""

    budget: np.ndarray  # (n_agents,) remaining budget after this step
    prev_budget: np.ndarray  # (n_agents,) budget before this step
    concession: np.ndarray  # (n_agents,) this step's concession, undiscounted
    t: int  # step index within the episode, from 0
    discount: float  # gamma ** t
    done: bool  # episode ends after this step (terminated or truncated)


@dataclass(frozen=True)
class BudgetRule:
    fn: Callable[[BudgetStep], object]
    common: bool


BUDGET_RULES = {}


def budget_rule(name, common):
    def register(fn):
        assert name not in BUDGET_RULES, f"budget rule '{name}' registered twice"
        BUDGET_RULES[name] = BudgetRule(fn, common)
        return fn

    return register


def get_rule(name):
    if name not in BUDGET_RULES:
        raise ValueError(
            f"Unknown budget_rule '{name}' (available: {sorted(BUDGET_RULES)})"
        )
    return BUDGET_RULES[name]


@budget_rule("joint_overspend", common=True)
def joint_overspend(s):
    """While any agent is overspent, every step pays the total overspend
    (negative); otherwise the leftover budget is paid out when the episode ends.
    With exact deltas (>= 0) an overspent budget stays negative; a learned
    estimate may dip below 0 and let a budget recover slightly."""
    if (s.budget < 0).any():
        return np.minimum(s.budget, 0).sum() / s.discount
    if s.done:
        return s.budget.sum() / s.discount
    return 0.0
