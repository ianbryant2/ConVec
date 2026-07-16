# ported from OptimisticExploration (WQMIX) modules/agents/central_rnn_agent.py,
# adapted to this repo's agent conventions (hidden_dim, use_rnn)

import torch.nn as nn
import torch as th
import torch.nn.functional as F


class CentralRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CentralRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions * args.central_action_embed)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        q = q.reshape(-1, self.args.n_actions, self.args.central_action_embed)
        return q, h


class CentralRNNNSAgent(nn.Module):
    """Non-shared central agent: one CentralRNNAgent per agent (no parameter sharing),
    mirroring RNNNSAgent."""

    def __init__(self, input_shape, args):
        super(CentralRNNNSAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.input_shape = input_shape
        self.agents = th.nn.ModuleList(
            [CentralRNNAgent(input_shape, args) for _ in range(self.n_agents)]
        )

    def init_hidden(self):
        # make hidden states on same device as model
        return th.cat([a.init_hidden() for a in self.agents])

    def forward(self, inputs, hidden_state):
        hiddens = []
        qs = []
        inputs = inputs.view(-1, self.n_agents, self.input_shape)
        for i in range(self.n_agents):
            q, h = self.agents[i](inputs[:, i], hidden_state[:, i])
            hiddens.append(h.unsqueeze(1))
            qs.append(q.unsqueeze(1))
        # (bs, n_agents, n_actions, central_action_embed) -> flatten agent dim so the
        # controller's .view(bs, n_agents, n_actions, -1) reshapes it back
        qs = th.cat(qs, dim=1).view(-1, self.args.n_actions, self.args.central_action_embed)
        return qs, th.cat(hiddens, dim=1)

    def cuda(self, device="cuda:0"):
        for a in self.agents:
            a.cuda(device=device)
