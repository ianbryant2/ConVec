"""OptimQAdditiveLinear: OptimQ with the joint action as a critic input.

    Q*(s, a) = head(relu(f(s) + sum_i g_i(a_i)))    -> (n_agents,)

Joint actions sharing an agent's action share its embedding g_i, so each
sample informs many joint actions and the max is less inflated by rarely seen
ones. Everything else is OptimQ's. budget.concession=optimq_additive_linear.
"""
import copy

import torch as th
import torch.nn as nn

from components.optimq import OptimQ


class OptimQAdditiveLinear(OptimQ):
    def _make_critic(self, input_dim, hidden_dim, n_joint, lr, layer_norm):
        return _AdditiveLinearCritic(
            input_dim, hidden_dim, self.n_agents, self.n_actions, self.joint_actions,
            lr, layer_norm, self.device,
        )


class _AdditiveLinearNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_agents, n_actions, joint_actions, layer_norm):
        super().__init__()
        norm = (lambda: [nn.LayerNorm(hidden_dim)]) if layer_norm else (lambda: [])
        # f(s): the state's embedding, left before its last ReLU so the action
        # embeddings are added to it first.
        self.state = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), *norm(), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), *norm(),
        )
        # g_i(a_i): row i * n_actions + a of the table is agent i's action a.
        self.action = nn.Embedding(n_agents * n_actions, hidden_dim)
        self.head = nn.Linear(hidden_dim, n_agents)
        offsets = th.arange(n_agents, device=joint_actions.device) * n_actions
        self.register_buffer("joint_rows", joint_actions + offsets)  # (n_joint, n_agents)

    def forward(self, inputs):
        """(B, input_dim) -> (B, n_agents, n_joint)"""
        f = self.state(inputs)  # (B, hidden)
        g = self.action(self.joint_rows).sum(1)  # (n_joint, hidden)
        q = self.head(th.relu(f.unsqueeze(1) + g))  # (B, n_joint, n_agents)
        return q.transpose(1, 2)

    def values_at(self, inputs, joint):
        """Agent i's value at its joint action: joint (B, n_agents, 1) -> (B, n_agents, 1)."""
        f = self.state(inputs)  # (B, hidden)
        g = self.action(self.joint_rows[joint.squeeze(-1)]).sum(-2)  # (B, n_agents, hidden)
        h = th.relu(f.unsqueeze(1) + g)  # (B, n_agents, hidden)
        # Agent i's head row applied to its own joint action's hidden vector only.
        q = (h * self.head.weight).sum(-1) + self.head.bias  # (B, n_agents)
        return q.unsqueeze(-1)


class _AdditiveLinearCritic:
    """Same interface as OptimQ's _Critic: online net, soft-updated target net, optimiser."""

    def __init__(self, input_dim, hidden_dim, n_agents, n_actions, joint_actions, lr,
                 layer_norm, device):
        self.net = _AdditiveLinearNet(
            input_dim, hidden_dim, n_agents, n_actions, joint_actions, layer_norm
        ).to(device)
        self.target_net = copy.deepcopy(self.net)
        self.optimiser = th.optim.Adam(self.net.parameters(), lr=lr)

    def q(self, net, inputs):
        """(B, input_dim) -> (B, n_agents, n_joint)"""
        return net(inputs)

    def q_at(self, net, inputs, joint):
        """Values at joint (B, n_agents, 1) only -> (B, n_agents, 1)."""
        return net.values_at(inputs, joint)
