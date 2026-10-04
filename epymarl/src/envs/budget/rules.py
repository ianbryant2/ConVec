"""Budget reward rules: turn the budget after a step into the agents' reward.

A rule with common=True returns one shared float, common=False one reward per
agent; common_reward must match. To add a rule:

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
    """While anyone is overspent, every step pays the total overspend;
    otherwise the leftover budget is paid at the end."""
    if (s.budget < 0).any():
        return np.minimum(s.budget, 0).sum() / s.discount
    if s.done:
        return s.budget.sum() / s.discount
    return 0.0


@budget_rule("joint_overspend_increment", common=True)
def joint_overspend_increment(s):
    """joint_overspend, but each unit of overspend is charged once, at the
    step that causes it (writeup Remark 24)."""
    if (s.budget < 0).any():
        increment = np.minimum(s.budget, 0) - np.minimum(s.prev_budget, 0)
        return increment.sum() / s.discount
    if s.done:
        return s.budget.sum() / s.discount
    return 0.0


@budget_rule("joint_budget_increment", common=True)
def joint_budget_increment(s):
    """Every step pays the change in total budget; no extra overspend penalty."""
    increment = (s.budget - s.prev_budget).sum()
    return increment / s.discount


@budget_rule("joint_budget_signed_increment", common=True)
def joint_budget_signed_increment(s):
    """While anyone is overspent, pays the change in overspend; otherwise
    pays the total remaining budget every step."""
    if (s.budget < 0).any():
        increment = np.minimum(s.budget, 0) - np.minimum(s.prev_budget, 0)
        return increment.sum() / s.discount
    return s.budget.sum() / s.discount


def overspend_potential(budget):
    """Total budget if everyone is within it, else the total overspend."""
    if (budget >= 0).all():
        return budget.sum()
    return np.minimum(budget, 0).sum()


@budget_rule("joint_overspend_potential", common=True)
def joint_overspend_potential(s):
    """joint_overspend_increment's return paid as it accrues: each step pays
    the change in overspend_potential."""
    reward = overspend_potential(s.budget) - overspend_potential(s.prev_budget)
    return reward / s.discount