"""OptimQ: learned joint-action critics Q*_i for the concession budget.

For each agent i, Q*_i(s, a) is the optimal action-value of agent i's own reward
r_i when the joint action a is treated as a single action (Definition 2 of the
writeup): the value agent i would get with control of every agent. The
concession is delta_i = max_a Q*_i(s, a) - Q*_i(s, a_t) >= 0 (Definition 4).

Taking the max over one noisy estimate biases delta upwards, so OptimQ trains
two independent critics Q1 and Q2, each on its own random half of the
transitions (as in double Q-learning), and uses the double estimator

    delta = 1/2 [Q2(s, argmax_a Q1(s, a)) - Q2(s, a_t)]
          + 1/2 [Q1(s, argmax_a Q2(s, a)) - Q1(s, a_t)]

with the argmax over available joint actions and each critic's slow-moving
target net as Q1/Q2. delta is not clamped at 0: the true delta is >= 0, but
clamping an estimate turns its zero-mean noise into a positive charge at every
step whose true delta is ~0, and over an episode those charges add up to the
size of real concessions. Unclamped, the noise cancels.

The critics also see the step index as t / time_limit. Episodes that are cut
off by the time limit end without a terminal state, and a critic that cannot
tell how many steps remain cannot value that; with t as an input, treating the
cut-off as terminal gives the finite-horizon Q*_i(s, t, a).

Each critic trains as in FitUtopianCritics of Algorithm 1: off-policy from its
replay buffer of (s, t, a, r, s', done) transitions with the original
per-agent rewards, double-Q targets (online net selects the next joint action,
target net evaluates it) and a soft-updated target net.

The networks output one value per joint action for every agent, so their size
grows as n_actions ** n_agents.
"""
import copy
import itertools

import torch as th
import torch.nn as nn
import torch.nn.functional as F

MAX_JOINT_ACTIONS = 100_000


class OptimQ:
    def __init__(
        self,
        args,
        state_dim,
        hidden_dim,
        lr,
        buffer_size,
        batch_size,
        updates_per_run,
        target_tau,
        grad_clip,
    ):
        # Defaults live in default.yaml (budget.concession_args.optimq).
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.gamma = args.gamma
        self.device = args.device
        # Budget runs are gymma envs, whose episodes end at time_limit.
        self.horizon = args.env_args["time_limit"]

        n_joint = self.n_actions**self.n_agents
        if n_joint > MAX_JOINT_ACTIONS:
            raise ValueError(
                f"OptimQ enumerates joint actions: {self.n_actions}^{self.n_agents} "
                f"= {n_joint} is more than {MAX_JOINT_ACTIONS}"
            )
        # joint_actions[j] is the per-agent action tuple of joint action j.
        self.joint_actions = th.tensor(
            list(itertools.product(range(self.n_actions), repeat=self.n_agents)),
            device=self.device,
        )
        self.radix = self.n_actions ** th.arange(
            self.n_agents - 1, -1, -1, device=self.device
        )

        self.critics = [
            _Critic(state_dim + 1, hidden_dim, self.n_agents, n_joint, lr, self.device)
            for _ in range(2)
        ]
        # Each transition trains only one critic, so their estimation errors are
        # independent; with shared data both would fit the same noisy averages and
        # the double estimator would reduce to the single one.
        self.buffers = [
            _TransitionBuffer(buffer_size // 2, self.device) for _ in self.critics
        ]
        self.batch_size = batch_size
        self.updates_per_run = updates_per_run
        self.target_tau = target_tau
        self.grad_clip = grad_clip

    def _inputs(self, states, t):
        """Critic input: the state with the step index t (B,) appended as t / horizon."""
        time = t.float().view(-1, 1) / self.horizon
        return th.cat([states.float(), time], dim=-1)

    def _joint_index(self, actions):
        """(B, n_agents) per-agent actions -> (B, n_agents, 1) joint index for gather."""
        joint = (actions.long() * self.radix).sum(-1)
        return joint.view(-1, 1, 1).expand(-1, self.n_agents, 1)

    def _joint_avail(self, avail_actions):
        """(B, n_agents, n_actions) per-agent mask -> (B, 1, n_joint) joint mask."""
        agents = th.arange(self.n_agents, device=self.device)
        per_agent = avail_actions[:, agents, self.joint_actions]  # (B, n_joint, n_agents)
        return per_agent.bool().all(-1).unsqueeze(1)

    @th.no_grad()
    def __call__(self, states, actions, avail_actions, t):
        """Concession delta (B, n_agents) for joint actions taken at states at step t."""
        inputs = self._inputs(states, th.full((len(states),), t, device=states.device))
        avail = self._joint_avail(avail_actions)
        taken = self._joint_index(actions)
        q1, q2 = (c.q(c.target_net, inputs) for c in self.critics)
        best1 = q1.masked_fill(~avail, -1e9).argmax(-1, keepdim=True)
        best2 = q2.masked_fill(~avail, -1e9).argmax(-1, keepdim=True)
        delta = 0.5 * (q2.gather(-1, best1) - q2.gather(-1, taken)) + 0.5 * (
            q1.gather(-1, best2) - q1.gather(-1, taken)
        )
        return delta.squeeze(-1)

    def update(self, transitions):
        """Split new transitions at random between the two critics' buffers and
        take updates_per_run gradient steps per critic.

        Returns {stat name: value} to log (the mean TD loss, and how far the two
        target critics disagree at the new transitions' joint actions), or None
        while a buffer is too small."""
        to_first = th.rand(len(transitions["state"]), device=self.device) < 0.5
        for buffer, rows in zip(self.buffers, (to_first, ~to_first)):
            if rows.any():
                buffer.add({k: v[rows] for k, v in transitions.items()})
        if min(len(b) for b in self.buffers) < self.batch_size:
            return None
        losses = [
            self._train_step(critic, buffer.sample(self.batch_size))
            for _ in range(self.updates_per_run)
            for critic, buffer in zip(self.critics, self.buffers)
        ]
        with th.no_grad():
            inputs = self._inputs(transitions["state"], transitions["t"])
            taken = self._joint_index(transitions["actions"])
            q1, q2 = (c.q(c.target_net, inputs).gather(-1, taken) for c in self.critics)
        return {
            "concession_loss": sum(losses) / len(losses),
            # Large gaps mean noisy Q* estimates, and so noisy deltas.
            "concession_critic_gap": (q1 - q2).abs().mean().item(),
        }

    def _train_step(self, critic, b):
        inputs = self._inputs(b["state"], b["t"])
        next_inputs = self._inputs(b["next_state"], b["t"] + 1)
        q_taken = critic.q(critic.net, inputs).gather(
            -1, self._joint_index(b["actions"])
        ).squeeze(-1)

        with th.no_grad():
            next_avail = self._joint_avail(b["next_avail"])
            # Terminal rows are masked out by (1 - done); keep their max finite.
            next_avail = next_avail | ~next_avail.any(-1, keepdim=True)
            next_q = critic.q(critic.net, next_inputs).masked_fill(~next_avail, -1e9)
            best_next = next_q.argmax(-1, keepdim=True)  # online net selects
            next_value = critic.q(critic.target_net, next_inputs).gather(-1, best_next)
            target = b["env_reward"] + self.gamma * (1 - b["terminated"]) * next_value.squeeze(-1)

        loss = F.mse_loss(q_taken, target)
        critic.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(critic.net.parameters(), self.grad_clip)
        critic.optimiser.step()

        with th.no_grad():
            for p, tp in zip(critic.net.parameters(), critic.target_net.parameters()):
                tp.mul_(1 - self.target_tau).add_(self.target_tau * p)
        return loss.item()


class _Critic:
    """One joint-action critic: online net, soft-updated target net, optimiser."""

    def __init__(self, state_dim, hidden_dim, n_agents, n_joint, lr, device):
        self.n_agents = n_agents
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents * n_joint),
        ).to(device)
        self.target_net = copy.deepcopy(self.net)
        self.optimiser = th.optim.Adam(self.net.parameters(), lr=lr)

    def q(self, net, inputs):
        """(B, input_dim) -> (B, n_agents, n_joint)"""
        return net(inputs).view(inputs.shape[0], self.n_agents, -1)


class _TransitionBuffer:
    """Fixed-size FIFO of transitions, stored as tensors on one device."""

    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.data = None
        self.pos = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, transitions):
        n = len(transitions["state"])
        if self.data is None:
            self.data = {
                k: th.zeros((self.capacity,) + v.shape[1:], dtype=v.dtype, device=self.device)
                for k, v in transitions.items()
            }
        idx = (self.pos + th.arange(n, device=self.device)) % self.capacity
        for k, v in transitions.items():
            self.data[k][idx] = v.to(self.device)
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idx = th.randint(self.size, (batch_size,), device=self.device)
        return {k: v[idx] for k, v in self.data.items()}
