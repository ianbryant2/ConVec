import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule
from utils.normalize import normalize
REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class SoftPoliciesSelector():

    def __init__(self, args):
        self.args = args
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        if test_mode and self.test_greedy:
            picked_actions = agent_inputs.max(dim=2)[1]
        else:
            m = Categorical(agent_inputs)
            picked_actions = m.sample().long()
        return picked_actions


REGISTRY["soft_policies"] = SoftPoliciesSelector

class OPTActionSelector():
    """Optimistic epsilon-greedy (ported from OptimisticExploration): on the epsilon
    branch, sample from a softmax over the optimistic estimator's values instead of
    uniformly at random."""

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.normal_weight = DecayThenFlatSchedule(args.normal_start, args.normal_finish, args.normal_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.max = args.normalmax

    def select_action(self, agent_inputs, opt_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)
        if self.args.use_temp_norm:
            self.normalmax = self.max - self.normal_weight.eval(t_env)
        else:
            self.normalmax = self.max

        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        masked_opt_values = opt_inputs.clone()
        if self.args.use_norm:
            masked_opt_values = normalize(masked_opt_values, avail_actions, 0, self.normalmax, dim=-1)
        masked_opt_values[avail_actions == 0.0] = -float("inf")  # should never be selected!
        opt_softmax = th.nn.functional.softmax(masked_opt_values, dim=-1)

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(opt_softmax).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]

        return picked_actions


REGISTRY["opt_selector"] = OPTActionSelector
