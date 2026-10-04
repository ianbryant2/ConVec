REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

# The stock runners with a concession budget (budget.*, --env-config=gymma_budget).
from .budget_runner import BudgetEpisodeRunner, BudgetParallelRunner
REGISTRY["budget_episode"] = BudgetEpisodeRunner
REGISTRY["budget_parallel"] = BudgetParallelRunner
