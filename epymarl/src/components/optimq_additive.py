"""OptimQ critics with the joint action as a critic input.

optimq_additive_linear:
    Q*(s, a) = head(relu(f(s) + sum_i g_i(a_i)))    -> (n_agents,)
optimq_additive_mlp (head_layers hidden layers after the sum):
    Q*(s, a) = head(mlp(relu(f(s) + sum_i g_i(a_i))))

Joint actions sharing an agent's action share its embedding g_i, so each
sample informs many joint actions and the max is less inflated by rarely seen
ones. The linear head combines the agents' actions only through that sum and
one ReLU; the MLP head's further layers (ReLU(W h + b)) let them interact
before the output. Everything else is OptimQ's.
"""
import copy

import torch as th
import torch.nn as nn

from components.optimq import OptimQ


class OptimQAdditiveLinear(OptimQ):
    head_layers = 0

    def _make_critic(self, input_dim, hidden_dim, n_joint, lr, layer_norm):
        return _AdditiveLinearCritic(
            input_dim, hidden_dim, self.n_agents, self.n_actions, self.joint_actions,
            lr, layer_norm, self.device, self.head_layers,
        )


class OptimQAdditiveMLP(OptimQAdditiveLinear):
    def __init__(self, args, state_dim, head_layers=1, **options):
        if head_layers < 1:
            raise ValueError(f"optimq_additive_mlp needs head_layers >= 1, got {head_layers}")
        self.head_layers = head_layers  # read by _make_critic, called from OptimQ.__init__
        super().__init__(args, state_dim, **options)


class _AdditiveLinearNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_agents, n_actions, joint_actions, layer_norm,
                 head_layers):
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
        # Hidden layers after relu(f(s) + sum_i g_i(a_i)), each ReLU(W h + b).
        self.mlp = nn.Sequential(*[
            m for _ in range(head_layers)
            for m in (nn.Linear(hidden_dim, hidden_dim), *norm(), nn.ReLU())
        ])
        self.head = nn.Linear(hidden_dim, n_agents)
        offsets = th.arange(n_agents, device=joint_actions.device) * n_actions
        self.register_buffer("joint_rows", joint_actions + offsets)  # (n_joint, n_agents)

    def _after_sum(self, h):
        # Nets saved before head_layers existed have no mlp.
        mlp = getattr(self, "mlp", None)
        return h if mlp is None else mlp(h)

    def forward(self, inputs):
        """(B, input_dim) -> (B, n_agents, n_joint)"""
        f = self.state(inputs)  # (B, hidden)
        g = self.action(self.joint_rows).sum(1)  # (n_joint, hidden)
        q = self.head(self._after_sum(th.relu(f.unsqueeze(1) + g)))  # (B, n_joint, n_agents)
        return q.transpose(1, 2)

    def values_at(self, inputs, joint):
        """Agent i's value at its joint action: joint (B, n_agents, 1) -> (B, n_agents, 1)."""
        f = self.state(inputs)  # (B, hidden)
        g = self.action(self.joint_rows[joint.squeeze(-1)]).sum(-2)  # (B, n_agents, hidden)
        h = self._after_sum(th.relu(f.unsqueeze(1) + g))  # (B, n_agents, hidden)
        # Agent i's head row applied to its own joint action's hidden vector only.
        q = (h * self.head.weight).sum(-1) + self.head.bias  # (B, n_agents)
        return q.unsqueeze(-1)


class _AdditiveLinearCritic:
    """Same interface as OptimQ's _Critic: online net, soft-updated target net, optimiser."""

    def __init__(self, input_dim, hidden_dim, n_agents, n_actions, joint_actions, lr,
                 layer_norm, device, head_layers):
        self.net = _AdditiveLinearNet(
            input_dim, hidden_dim, n_agents, n_actions, joint_actions, layer_norm, head_layers
        ).to(device)
        self.target_net = copy.deepcopy(self.net)
        self.optimiser = th.optim.Adam(self.net.parameters(), lr=lr)

    def q(self, net, inputs):
        """(B, input_dim) -> (B, n_agents, n_joint)"""
        return net(inputs)

    def q_at(self, net, inputs, joint):
        """Values at joint (B, n_agents, 1) only -> (B, n_agents, 1)."""
        return net.values_at(inputs, joint)
