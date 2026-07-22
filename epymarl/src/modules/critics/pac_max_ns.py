from einops import repeat
import torch as th
import torch.nn as nn

from modules.critics.mlp import MLP


class PACMaxCriticNS(nn.Module):
    """Learns f_i(state, a_i) ~= max_{a_-i} Q_i(state, a_-i, a_i).

    Same target quantity as pareto_ac's exact combinatorial max
    (`critic(batch, compute_all=True)[0].max(dim=3)`), but queried from just
    the full state and a single individual action instead of enumerating (or
    sampling) the other agents' joint actions at inference time.
    """

    def __init__(self, scheme, args):
        super(PACMaxCriticNS, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        self.critics = [
            MLP(input_shape, args.hidden_dim, 1) for _ in range(self.n_agents)
        ]

        self.device = "cuda" if args.use_cuda else "cpu"

    def forward(self, batch, t=None):
        """Returns (bs, max_t, n_agents, n_actions): one value per own action."""
        base, bs, max_t = self._build_base_inputs(batch, t=t)

        own_actions = th.eye(self.n_actions, device=self.device)
        own_actions = repeat(
            own_actions, "a f -> n s ag a f", n=bs, s=max_t, ag=self.n_agents
        )
        base = repeat(base, "n s ag f -> n s ag a f", a=self.n_actions)
        inputs = th.cat((base, own_actions), dim=-1)

        vs = []
        for i in range(self.n_agents):
            vs.append(self.critics[i](inputs[:, :, i]).squeeze(-1).unsqueeze(2))
        return th.cat(vs, dim=2)

    def _build_base_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        inputs = []
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        if self.args.obs_individual_obs:
            inputs.append(
                batch["obs"][:, ts]
                .view(bs, max_t, -1)
                .unsqueeze(2)
                .repeat(1, 1, self.n_agents, 1)
            )

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(
                    th.zeros_like(batch["actions_onehot"][:, 0:1])
                    .view(bs, max_t, 1, -1)
                    .repeat(1, 1, self.n_agents, 1)
                )
            elif isinstance(t, int):
                inputs.append(
                    batch["actions_onehot"][:, slice(t - 1, t)]
                    .view(bs, max_t, 1, -1)
                    .repeat(1, 1, self.n_agents, 1)
                )
            else:
                last_actions = th.cat(
                    [
                        th.zeros_like(batch["actions_onehot"][:, 0:1]),
                        batch["actions_onehot"][:, :-1],
                    ],
                    dim=1,
                )
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(
                    1, 1, self.n_agents, 1
                )
                inputs.append(last_actions)

        return th.cat(inputs, dim=-1), bs, max_t

    def _get_input_shape(self, scheme):
        input_shape = scheme["state"]["vshape"]
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"] * self.n_agents
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        input_shape += self.n_actions
        return input_shape

    def parameters(self):
        params = list(self.critics[0].parameters())
        for i in range(1, self.n_agents):
            params += list(self.critics[i].parameters())
        return params

    def state_dict(self):
        return [a.state_dict() for a in self.critics]

    def load_state_dict(self, state_dict):
        for i, a in enumerate(self.critics):
            a.load_state_dict(state_dict[i])

    def cuda(self):
        for c in self.critics:
            c.cuda()
